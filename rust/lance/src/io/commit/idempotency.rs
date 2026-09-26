// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Application-level idempotency keys for concurrent commits.
//!
//! A writer that must apply each unit of work exactly once (for example, a
//! queue consumer whose messages can be redelivered to another consumer)
//! stamps the ids of that work into one transaction property as a JSON array
//! of strings. When the commit is configured with the property name, every
//! transaction committed between our read version and our target version is
//! checked for an overlapping key, and an overlap fails the commit with
//! [`Error::DuplicateTransaction`] instead of rebasing over it.
//!
//! Coverage: `commit_transaction` loads every version after the read version
//! before each attempt, and the manifest put is create-if-absent, so a
//! successful commit has checked every version between its read version and
//! the version it created. Versions at or before the read version are the
//! caller's responsibility.

use std::collections::BTreeSet;

use lance_core::{Error, Result};

use crate::dataset::transaction::Transaction;

/// The idempotency keys of the transaction being committed.
#[derive(Debug, Clone)]
pub(crate) struct IdempotencyKeys {
    property: String,
    keys: BTreeSet<String>,
}

impl IdempotencyKeys {
    /// Read our transaction's keys from `property`.
    ///
    /// Fails when the property is missing or is not a JSON array of strings:
    /// the caller asked for the check, so a transaction without keys would
    /// silently commit unchecked.
    pub(crate) fn from_transaction(property: &str, transaction: &Transaction) -> Result<Self> {
        let raw = transaction
            .transaction_properties
            .as_ref()
            .and_then(|properties| properties.get(property))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "idempotency property '{property}' is not set on the transaction being committed"
                ))
            })?;
        Ok(Self {
            property: property.to_string(),
            keys: parse_keys(property, raw)?,
        })
    }

    /// Fail when `other`, committed at `other_version`, carries any of our keys.
    ///
    /// A transaction without the property (compaction, index builds, writers
    /// that do not use idempotency keys) carries no keys. A malformed value
    /// fails the commit, since it cannot be checked.
    pub(crate) fn check(&self, other: &Transaction, other_version: u64) -> Result<()> {
        let Some(raw) = other
            .transaction_properties
            .as_ref()
            .and_then(|properties| properties.get(&self.property))
        else {
            return Ok(());
        };
        let theirs = parse_keys(&self.property, raw)?;
        let overlap: Vec<String> = self.keys.intersection(&theirs).cloned().collect();
        if overlap.is_empty() {
            Ok(())
        } else {
            Err(Error::duplicate_transaction(other_version, overlap))
        }
    }
}

fn parse_keys(property: &str, raw: &str) -> Result<BTreeSet<String>> {
    serde_json::from_str::<Vec<String>>(raw)
        .map(|keys| keys.into_iter().collect())
        .map_err(|err| {
            Error::invalid_input(format!(
                "idempotency property '{property}' is not a JSON array of strings: {err}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::dataset::transaction::Operation;

    const PROPERTY: &str = "idempotency_keys";

    fn transaction(properties: &[(&str, &str)]) -> Transaction {
        let mut txn = Transaction::new(1, Operation::Append { fragments: vec![] }, None);
        txn.transaction_properties = Some(Arc::new(
            properties
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<HashMap<_, _>>(),
        ));
        txn
    }

    #[test]
    fn overlapping_keys_fail_with_the_overlap() {
        let ours = IdempotencyKeys::from_transaction(
            PROPERTY,
            &transaction(&[(PROPERTY, r#"["a","b","c"]"#)]),
        )
        .unwrap();
        let err = ours
            .check(&transaction(&[(PROPERTY, r#"["c","d","a"]"#)]), 7)
            .unwrap_err();
        match err {
            Error::DuplicateTransaction { version, keys, .. } => {
                assert_eq!(version, 7);
                assert_eq!(keys, vec!["a".to_string(), "c".to_string()]);
            }
            other => panic!("expected DuplicateTransaction, got {other:?}"),
        }
    }

    #[test]
    fn disjoint_and_unkeyed_transactions_pass() {
        let ours =
            IdempotencyKeys::from_transaction(PROPERTY, &transaction(&[(PROPERTY, r#"["a"]"#)]))
                .unwrap();
        ours.check(&transaction(&[(PROPERTY, r#"["b"]"#)]), 2)
            .unwrap();
        ours.check(&transaction(&[("other", "x")]), 3).unwrap();
        let mut bare = transaction(&[]);
        bare.transaction_properties = None;
        ours.check(&bare, 4).unwrap();
    }

    #[test]
    fn missing_or_malformed_keys_fail_closed() {
        assert!(IdempotencyKeys::from_transaction(PROPERTY, &transaction(&[])).is_err());
        assert!(
            IdempotencyKeys::from_transaction(PROPERTY, &transaction(&[(PROPERTY, "a,b")]))
                .is_err()
        );
        let ours =
            IdempotencyKeys::from_transaction(PROPERTY, &transaction(&[(PROPERTY, r#"["a"]"#)]))
                .unwrap();
        assert!(
            ours.check(&transaction(&[(PROPERTY, r#"{"a":1}"#)]), 5)
                .is_err()
        );
    }

    /// End-to-end commits through `CommitBuilder` on a local dataset.
    mod commit {
        use std::sync::Arc;

        use arrow_array::{Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_core::utils::tempfile::TempStrDir;
        use lance_io::object_store::ObjectStore;
        use lance_table::format::{IndexMetadata, Manifest};
        use lance_table::io::commit::{
            CommitError, CommitHandler, ConditionalPutCommitHandler, ManifestLocation,
            ManifestNamingScheme, ManifestWriter,
        };
        use object_store::path::Path;
        use tokio::sync::Notify;

        use super::PROPERTY;
        use crate::Dataset;
        use crate::dataset::transaction::Transaction;
        use crate::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
        use lance_core::{Error, Result};

        fn schema() -> Arc<ArrowSchema> {
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "x",
                DataType::Int32,
                false,
            )]))
        }

        fn batch(values: &[i32]) -> RecordBatch {
            RecordBatch::try_new(schema(), vec![Arc::new(Int32Array::from(values.to_vec()))])
                .unwrap()
        }

        async fn create(uri: &str) -> Dataset {
            InsertBuilder::new(uri)
                .execute(vec![batch(&[0])])
                .await
                .unwrap()
        }

        /// An uncommitted append read at `read_version`, carrying `keys`.
        async fn staged(
            uri: &str,
            read_version: u64,
            values: &[i32],
            keys: &[&str],
        ) -> Transaction {
            let dataset = Dataset::open(uri)
                .await
                .unwrap()
                .checkout_version(read_version)
                .await
                .unwrap();
            let params = WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            };
            let mut txn = InsertBuilder::new(Arc::new(dataset))
                .with_params(&params)
                .execute_uncommitted(vec![batch(values)])
                .await
                .unwrap();
            assert_eq!(txn.read_version, read_version);
            if !keys.is_empty() {
                let json = serde_json::to_string(keys).unwrap();
                txn.transaction_properties = Some(Arc::new(
                    [(PROPERTY.to_string(), json)].into_iter().collect(),
                ));
            }
            txn
        }

        async fn commit_keyed(uri: &str, txn: Transaction, max_retries: u32) -> Result<Dataset> {
            CommitBuilder::new(uri)
                .with_idempotency_property(PROPERTY)
                .with_max_retries(max_retries)
                .execute(txn)
                .await
        }

        async fn values(uri: &str) -> Vec<i32> {
            let dataset = Dataset::open(uri).await.unwrap();
            let table = dataset.scan().try_into_batch().await.unwrap();
            let mut values = table
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec();
            values.sort();
            values
        }

        fn duplicate(result: Result<Dataset>) -> (u64, Vec<String>) {
            match result {
                Err(Error::DuplicateTransaction { version, keys, .. }) => (version, keys),
                Err(other) => panic!("expected DuplicateTransaction, got {other:?}"),
                Ok(ds) => panic!(
                    "expected DuplicateTransaction, committed v{}",
                    ds.version().version
                ),
            }
        }

        #[tokio::test]
        async fn overlap_after_read_version_fails_instead_of_rebasing() {
            let dir = TempStrDir::default();
            let uri = dir.as_str();
            create(uri).await;
            let first = staged(uri, 1, &[1, 2], &["a", "b"]).await;
            let second = staged(uri, 1, &[2, 3], &["b", "c"]).await;

            assert_eq!(
                commit_keyed(uri, first, 0).await.unwrap().version().version,
                2
            );
            let (version, keys) = duplicate(commit_keyed(uri, second, 0).await);

            assert_eq!((version, keys), (2, vec!["b".to_string()]));
            assert_eq!(values(uri).await, vec![0, 1, 2]);
        }

        #[tokio::test]
        async fn disjoint_and_unkeyed_versions_are_rebased_over() {
            let dir = TempStrDir::default();
            let uri = dir.as_str();
            create(uri).await;
            let keyed = staged(uri, 1, &[1], &["a"]).await;
            let unkeyed = staged(uri, 1, &[2], &[]).await;
            let ours = staged(uri, 1, &[3], &["b"]).await;

            commit_keyed(uri, keyed, 0).await.unwrap();
            CommitBuilder::new(uri).execute(unkeyed).await.unwrap();
            let landed = commit_keyed(uri, ours, 0).await.unwrap();

            assert_eq!(landed.version().version, 4);
            assert_eq!(values(uri).await, vec![0, 1, 2, 3]);
        }

        #[tokio::test]
        async fn retrying_a_commit_that_landed_reports_its_own_version() {
            let dir = TempStrDir::default();
            let uri = dir.as_str();
            create(uri).await;
            let txn = staged(uri, 1, &[1], &["a", "b"]).await;

            commit_keyed(uri, txn.clone(), 0).await.unwrap();
            let (version, keys) = duplicate(commit_keyed(uri, txn, 5).await);

            assert_eq!((version, keys), (2, vec!["a".to_string(), "b".to_string()]));
            assert_eq!(values(uri).await, vec![0, 1]);
        }

        #[tokio::test]
        async fn missing_property_on_our_transaction_fails() {
            let dir = TempStrDir::default();
            let uri = dir.as_str();
            create(uri).await;
            let txn = staged(uri, 1, &[1], &[]).await;

            let err = commit_keyed(uri, txn, 0).await.unwrap_err();

            assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        }

        #[tokio::test]
        async fn without_the_option_an_overlap_is_rebased_over() {
            let dir = TempStrDir::default();
            let uri = dir.as_str();
            create(uri).await;
            let first = staged(uri, 1, &[1, 2], &["a", "b"]).await;
            let second = staged(uri, 1, &[2, 3], &["b", "c"]).await;

            CommitBuilder::new(uri).execute(first).await.unwrap();
            let landed = CommitBuilder::new(uri).execute(second).await.unwrap();

            assert_eq!(landed.version().version, 3);
            assert_eq!(values(uri).await, vec![0, 1, 2, 2, 3]);
        }

        /// Parks the first manifest put until the test lets it go, so a rival
        /// can create the same version after the key check ran.
        #[derive(Debug, Default)]
        struct ParkFirstPut {
            reached: Notify,
            release: Notify,
            parked: std::sync::atomic::AtomicBool,
        }

        #[async_trait::async_trait]
        impl CommitHandler for ParkFirstPut {
            async fn commit(
                &self,
                manifest: &mut Manifest,
                indices: Option<Vec<IndexMetadata>>,
                base_path: &Path,
                object_store: &ObjectStore,
                manifest_writer: ManifestWriter,
                naming_scheme: ManifestNamingScheme,
                transaction: Option<lance_table::format::Transaction>,
            ) -> std::result::Result<ManifestLocation, CommitError> {
                if !self.parked.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    self.reached.notify_one();
                    self.release.notified().await;
                }
                ConditionalPutCommitHandler
                    .commit(
                        manifest,
                        indices,
                        base_path,
                        object_store,
                        manifest_writer,
                        naming_scheme,
                        transaction,
                    )
                    .await
            }
        }

        #[tokio::test]
        async fn rival_landing_after_the_check_is_caught_on_the_retry() {
            let dir = TempStrDir::default();
            let uri = dir.as_str().to_string();
            create(&uri).await;
            let ours = staged(&uri, 1, &[1, 2], &["a", "b"]).await;
            let rival = staged(&uri, 1, &[2, 3], &["b", "c"]).await;
            let handler = Arc::new(ParkFirstPut::default());

            let task = {
                let (uri, handler) = (uri.clone(), handler.clone());
                tokio::spawn(async move {
                    CommitBuilder::new(uri.as_str())
                        .with_idempotency_property(PROPERTY)
                        .with_max_retries(5)
                        .with_commit_handler(handler)
                        .execute(ours)
                        .await
                })
            };
            // Our check ran against an empty range and our put of v2 is parked.
            handler.reached.notified().await;
            let rival_landed = CommitBuilder::new(uri.as_str())
                .execute(rival)
                .await
                .unwrap();
            assert_eq!(rival_landed.version().version, 2);
            handler.release.notify_one();

            let (version, keys) = duplicate(task.await.unwrap());
            assert_eq!((version, keys), (2, vec!["b".to_string()]));
            assert_eq!(values(&uri).await, vec![0, 2, 3]);
        }
    }
}
