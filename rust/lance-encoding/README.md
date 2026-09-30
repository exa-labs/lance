# lance-encoding

`lance-encoding` is an internal sub-crate, containing encoders and decoders
for the Lance file format.

**Important Note**: This crate is **not intended for external usage**.

Large per-value Zstd/LZ4 pages use byte-balanced groups of whole values, with
independent codec contexts and ordered assembly of the compressed bytes and offsets.
The per-value frames and file format are unchanged. Pages below 4 MiB and pages
with only one value use the serial path.

Scoped compression helpers share a process-wide, nonblocking budget of
`get_num_compute_intensive_cpus() - 1` threads, in addition to the existing
encoding callers. This count follows `LANCE_CPU_THREADS` or the detected CPU
count minus the I/O reservation. Setting `LANCE_CPU_THREADS=1` disables the
helpers. Busy callers keep compressing serially rather than waiting for a helper;
no encoding task waits for nested work on its own CPU pool. Parallel assembly
temporarily retains compressed group buffers before copying them into the page.
