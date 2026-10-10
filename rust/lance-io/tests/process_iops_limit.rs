// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The application default for the process-wide IOPS limit is process-global
//! state, so it is tested in its own test binary, in a single test.

use std::{sync::Arc, time::Duration};

use lance_core::utils::tempfile::TempObjFile;
use lance_io::{
    object_store::ObjectStore,
    scheduler::{ScanScheduler, SchedulerConfig},
    set_default_process_iops_limit,
    utils::CachedFileSize,
};

#[tokio::test]
async fn test_limit_is_settable_until_first_io() {
    let tmp_file = TempObjFile::default();
    let obj_store = Arc::new(ObjectStore::local());
    let content = vec![7u8; 4096];
    obj_store.put(&tmp_file, &content).await.unwrap();

    // A scheduler whose I/O loop is running but has no requests yet.
    let scheduler = ScanScheduler::new(obj_store, SchedulerConfig::default_for_testing());
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(set_default_process_iops_limit(256));

    let file_scheduler = scheduler
        .open_file(&tmp_file, &CachedFileSize::unknown())
        .await
        .unwrap();
    #[allow(clippy::single_range_in_vec_init)]
    let data = file_scheduler
        .submit_request(vec![0..4096], 0)
        .await
        .unwrap();
    assert_eq!(data[0].as_ref(), content.as_slice());

    // The first I/O fixed the limit.
    assert!(!set_default_process_iops_limit(512));
}
