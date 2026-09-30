// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded, byte-balanced compression of independent values within a page.

use std::{ops::Range, sync::LazyLock, thread};

use arrow_buffer::ArrowNativeType;
use lance_core::{Error, Result, utils::tokio::get_num_compute_intensive_cpus};
use tokio::sync::{Semaphore, SemaphorePermit};

use super::{CompressionConfig, CompressionScheme, GeneralBufferCompressor};

// Amortize scoped-thread startup and an independent codec context.
const MIN_BYTES_PER_WORKER: usize = 2 * 1024 * 1024;

// At most CPU_THREADS - 1 extra threads across all pages, in addition to encoding callers.
static HELPERS: LazyLock<Semaphore> =
    LazyLock::new(|| Semaphore::new(get_num_compute_intensive_cpus().saturating_sub(1)));

struct CompressedChunk {
    bytes: Vec<u8>,
    ends: Vec<usize>,
}

fn ranges<T: ArrowNativeType>(offsets: &[T], workers: usize) -> Vec<Range<usize>> {
    let values = offsets.len() - 1;
    let first = offsets[0].as_usize();
    let bytes = offsets[values].as_usize() - first;
    let mut ranges = Vec::with_capacity(workers);
    let mut start = 0;
    for worker in 1..workers {
        let target = first + bytes / workers * worker;
        let end = offsets.partition_point(|offset| offset.as_usize() < target);
        if end > start && end < values {
            ranges.push(start..end);
            start = end;
        }
    }
    ranges.push(start..values);
    ranges
}

fn compress_chunk<T: ArrowNativeType>(
    data: &[u8],
    offsets: &[T],
    config: CompressionConfig,
) -> Result<CompressedChunk> {
    let compressor = GeneralBufferCompressor::get_compressor(config)?;
    let mut bytes =
        Vec::with_capacity(offsets[offsets.len() - 1].as_usize() - offsets[0].as_usize());
    let mut ends = Vec::with_capacity(offsets.len() - 1);
    for pair in offsets.windows(2) {
        compressor.compress(&data[pair[0].as_usize()..pair[1].as_usize()], &mut bytes)?;
        ends.push(bytes.len());
    }
    Ok(CompressedChunk { bytes, ends })
}

fn compress_parallel<T: ArrowNativeType>(
    data: &[u8],
    offsets: &[T],
    compressed: &mut Vec<u8>,
    config: CompressionConfig,
    ranges: &[Range<usize>],
) -> Result<Vec<T>> {
    let chunks = thread::scope(|scope| -> Result<Vec<CompressedChunk>> {
        let mut handles = Vec::with_capacity(ranges.len() - 1);
        for range in &ranges[1..] {
            let offsets = &offsets[range.start..=range.end];
            handles.push(
                thread::Builder::new()
                    .name("lance-compress".into())
                    .spawn_scoped(scope, move || compress_chunk(data, offsets, config))?,
            );
        }
        let first = &ranges[0];
        let mut chunks = vec![compress_chunk(
            data,
            &offsets[first.start..=first.end],
            config,
        )?];
        for handle in handles {
            chunks.push(
                handle
                    .join()
                    .map_err(|_| Error::internal("Compression worker panicked"))??,
            );
        }
        Ok(chunks)
    })?;

    let mut new_offsets = Vec::with_capacity(offsets.len());
    new_offsets.push(T::from_usize(0).ok_or_else(|| Error::invalid_input("Invalid offset type"))?);
    for chunk in chunks {
        let base = compressed.len();
        for end in chunk.ends {
            let end = base
                .checked_add(end)
                .and_then(T::from_usize)
                .ok_or_else(|| Error::invalid_input("Compressed value offset overflow"))?;
            new_offsets.push(end);
        }
        compressed.extend_from_slice(&chunk.bytes);
    }
    Ok(new_offsets)
}

pub(super) fn try_compress<T: ArrowNativeType>(
    data: &[u8],
    offsets: &[T],
    compressed: &mut Vec<u8>,
    config: CompressionConfig,
) -> Result<Option<Vec<T>>> {
    if offsets.len() < 3
        || !matches!(
            config.scheme,
            CompressionScheme::Zstd | CompressionScheme::Lz4
        )
    {
        return Ok(None);
    }
    let end = offsets[offsets.len() - 1].as_usize();
    let bytes = end
        .checked_sub(offsets[0].as_usize())
        .ok_or_else(|| Error::invalid_input("Value offsets are not monotonic"))?;
    if bytes < 2 * MIN_BYTES_PER_WORKER {
        return Ok(None);
    }
    if end > data.len()
        || offsets
            .windows(2)
            .any(|pair| pair[0].as_usize() > pair[1].as_usize())
    {
        return Err(Error::invalid_input(
            "Value offsets exceed the data or are not monotonic",
        ));
    }
    try_compress_with_budget(data, offsets, compressed, config, &HELPERS)
}

fn try_compress_with_budget<T: ArrowNativeType>(
    data: &[u8],
    offsets: &[T],
    compressed: &mut Vec<u8>,
    config: CompressionConfig,
    helpers: &Semaphore,
) -> Result<Option<Vec<T>>> {
    let Some((_permit, ranges)) = reserve_helpers(offsets, helpers) else {
        return Ok(None);
    };
    compress_parallel(data, offsets, compressed, config, &ranges).map(Some)
}

fn reserve_helpers<'a, T: ArrowNativeType>(
    offsets: &[T],
    helpers: &'a Semaphore,
) -> Option<(SemaphorePermit<'a>, Vec<Range<usize>>)> {
    let bytes = offsets[offsets.len() - 1].as_usize() - offsets[0].as_usize();
    let desired = (bytes / MIN_BYTES_PER_WORKER)
        .min(offsets.len() - 1)
        .saturating_sub(1)
        .min(helpers.available_permits())
        .min(u32::MAX as usize) as u32;
    (1..=desired).rev().find_map(|count| {
        let ranges = ranges(offsets, count as usize + 1);
        if ranges.len() < 2 {
            return None;
        }
        let permit = helpers.try_acquire_many((ranges.len() - 1) as u32).ok()?;
        Some((permit, ranges))
    })
}

#[cfg(all(test, feature = "zstd", feature = "lz4"))]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(vec![0u64, 2, 4, 6, 8], 4, vec![0..1, 1..2, 2..3, 3..4])]
    #[case(vec![7u64, 7, 7, 207, 208, 208], 4, vec![0..3, 3..5])]
    #[case(vec![0u64, 0, 0, 0], 4, vec![0..3])]
    fn partitions_cover_values_in_order(
        #[case] offsets: Vec<u64>,
        #[case] workers: usize,
        #[case] expected: Vec<Range<usize>>,
    ) {
        assert_eq!(ranges(&offsets, workers), expected);
    }

    #[rstest]
    #[case(CompressionScheme::Zstd, Some(0))]
    #[case(CompressionScheme::Zstd, Some(1))]
    #[case(CompressionScheme::Zstd, Some(9))]
    #[case(CompressionScheme::Lz4, None)]
    fn parallel_bytes_match_serial(#[case] scheme: CompressionScheme, #[case] level: Option<i32>) {
        fn check<T: ArrowNativeType>(config: CompressionConfig) {
            let mut data = b"unused prefix".to_vec();
            let mut offsets = vec![T::from_usize(data.len()).unwrap()];
            for len in [0, 1, 31, 200_000, 0, 7, 80_000, 3, 0] {
                data.extend((0..len).map(|i| ((i * 179 + i / 31) % 256) as u8));
                offsets.push(T::from_usize(data.len()).unwrap());
            }
            let compressor = GeneralBufferCompressor::get_compressor(config).unwrap();
            let mut expected = b"existing output".to_vec();
            let mut expected_offsets = vec![T::from_usize(0).unwrap()];
            for pair in offsets.windows(2) {
                compressor
                    .compress(&data[pair[0].as_usize()..pair[1].as_usize()], &mut expected)
                    .unwrap();
                expected_offsets.push(T::from_usize(expected.len()).unwrap());
            }
            for workers in [1, 2, 4, 8] {
                let mut actual = b"existing output".to_vec();
                let actual_offsets = compress_parallel(
                    &data,
                    &offsets,
                    &mut actual,
                    config,
                    &ranges(&offsets, workers),
                )
                .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(actual_offsets, expected_offsets);
            }
        }
        let config = CompressionConfig::new(scheme, level);
        check::<u32>(config);
        check::<u64>(config);
    }

    #[test]
    fn busy_budget_uses_serial_path_and_releases_helpers() {
        let data = vec![42; 4 * MIN_BYTES_PER_WORKER];
        let offsets = [0u32, (2 * MIN_BYTES_PER_WORKER) as u32, data.len() as u32];
        let budget = Semaphore::new(1);
        let config = CompressionConfig::new(CompressionScheme::Lz4, None);
        let held = budget.try_acquire().unwrap();
        let mut output = Vec::new();
        assert!(
            try_compress_with_budget(&data, &offsets, &mut output, config, &budget)
                .unwrap()
                .is_none()
        );
        assert!(output.is_empty());
        drop(held);
        assert!(
            try_compress_with_budget(&data, &offsets, &mut output, config, &budget)
                .unwrap()
                .is_some()
        );
        assert_eq!(budget.available_permits(), 1);

        let invalid = CompressionConfig::new(CompressionScheme::Fsst, None);
        let error =
            try_compress_with_budget(&data, &offsets, &mut output, invalid, &budget).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("fsst is not usable"));
        assert_eq!(budget.available_permits(), 1);
    }

    #[test]
    fn small_or_single_value_pages_stay_serial() {
        let config = CompressionConfig::new(CompressionScheme::Lz4, None);
        for offsets in [vec![0u32], vec![0, 0], vec![0, 1, 2]] {
            assert!(
                try_compress(b"xy", &offsets, &mut Vec::new(), config)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn invalid_offsets_are_rejected_before_partitioning() {
        let config = CompressionConfig::new(CompressionScheme::Lz4, None);
        let data = vec![0; 4 * MIN_BYTES_PER_WORKER];
        for offsets in [
            vec![0u32, data.len() as u32 + 1, data.len() as u32],
            vec![0, 0, data.len() as u32 + 1],
        ] {
            let error = try_compress(&data, &offsets, &mut Vec::new(), config).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("Value offsets"));
        }
    }

    #[test]
    fn skewed_values_reserve_only_the_helpers_they_use() {
        let big = 10 * MIN_BYTES_PER_WORKER as u64;
        let offsets = [0u64, 0, big, big + 1, big + 2];
        let budget = Semaphore::new(3);
        let (permit, groups) = reserve_helpers(&offsets, &budget).unwrap();
        assert_eq!(groups, vec![0..2, 2..4]);
        assert_eq!(permit.num_permits(), 1);
        assert_eq!(budget.available_permits(), 2);
        drop(permit);
        assert_eq!(budget.available_permits(), 3);
    }
}
