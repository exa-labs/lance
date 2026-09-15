# Exa wheel release

Release wheels are built from a clean checkout with immutable tool inputs:

- Rust 1.91.0 (`f8297e351a40c1439a467bbbb6879088047f50b3`)
- Zig 0.16.0
- maturin 1.15.0
- protoc 24.4

The downloaded Zig, maturin, and protoc archives are verified by SHA-256 before
use. Cargo builds with `--locked`, the source commit timestamp controls archive
timestamps, and source paths are remapped. The reproduction command performs
two clean checkouts sequentially at the same canonical build path because Rust
ThinLTO output is sensitive to absolute build paths beyond diagnostic path
remapping.

From a clean Linux x86_64 checkout, reproduce each wheel twice:

```shell
.github/exa-release/reproduce-wheel.sh x86_64 /persistent/path/exa-wheel-x86_64
.github/exa-release/reproduce-wheel.sh aarch64 /persistent/path/exa-wheel-aarch64
```

The command fails unless the two clean builds have the same SHA-256 and each
artifact has:

- maturin 1.15.0 wheel metadata
- the expected architecture and manylinux 2.17 tag
- Clang 21.1.0, Rust 1.91.0, and LLD 21.1.0 provenance
- no `libgcc_s.so.1` dependency
- a maximum GLIBC symbol version of 2.17

Before publication, install the verified aarch64 wheel on an aarch64 host,
start the repository LocalStack service, and run:

```shell
wheel="$(find /persistent/path/exa-wheel-aarch64/verified -name '*.whl' -print -quit)"
uv run --isolated --python 3.10 --with "${wheel}" --with boto3 \
  python .github/exa-release/test-wheel-s3.py
```

This writes and repeatedly opens a LocalStack dataset, then requires repeated
HTTPS S3 requests to reach an AWS service response instead of failing during
DNS, TCP, or TLS setup. Publish only the wheel SHA-256 recorded by the
reproduction step; release assets are immutable.
