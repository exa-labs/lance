# Exa wheel release

Release wheels are built from a clean checkout with immutable tool inputs:

- Rust 1.91.0 (`f8297e351a40c1439a467bbbb6879088047f50b3`)
- Zig 0.16.0
- maturin 1.15.0
- protoc 24.4

The downloaded Zig, maturin, and protoc archives are verified by SHA-256 before
use. Cargo builds with `--locked`, the source commit timestamp controls archive
timestamps, and source paths are remapped.

## Build

Build one wheel per architecture from a clean checkout of the release tag.
Both architectures cross-build with `--zig` from a single Linux x86_64 host,
so the two builds can run in parallel. Each output directory must be empty.

```shell
git clone --branch <release-tag> https://github.com/exa-labs/lance.git /persistent/path/lance-release
cd /persistent/path/lance-release
.github/exa-release/build-wheel.sh /persistent/path/lance-release x86_64 /persistent/path/exa-wheel-x86_64 &
.github/exa-release/build-wheel.sh /persistent/path/lance-release aarch64 /persistent/path/exa-wheel-aarch64 &
wait
```

Each output directory then contains the wheel and `<wheel>.provenance.txt`,
which records the source commit, target, toolchain archive SHA-256s, Rust,
Zig, and maturin versions, and the `wheel_sha256`.

## Toolchain validation

`build-wheel.sh` runs `validate-wheel.sh` on the built wheel and fails unless
it passes. The validation inspects the wheel metadata and the ELF headers of
`lance/lance.abi3.so`, which is the proof that the pinned toolchain produced
the artifact:

- maturin 1.15.0 wheel metadata and the expected
  `cp310-abi3-manylinux_2_17_<arch>.manylinux2014_<arch>` wheel tag
- the expected ELF machine (`X86-64` or `AArch64`)
- `.comment` section provenance: Clang 21.1.0, Rust 1.91.0, and LLD 21.1.0
- `NEEDED` libraries limited to the dynamic loader, `libc.so.6`, `libdl.so.2`,
  `libm.so.6`, and `libpthread.so.0` (in particular, no `libgcc_s.so.1`)
- a maximum GLIBC symbol version of 2.17

To re-check a downloaded wheel, run
`.github/exa-release/validate-wheel.sh <wheel> <x86_64|aarch64>`.

## Smoke test

Before publication, install each wheel on a native host of its architecture,
start the repository LocalStack service, and run:

```shell
wheel="$(find /persistent/path/exa-wheel-aarch64 -maxdepth 1 -name '*.whl' -print -quit)"
uv run --isolated --python 3.10 --with "${wheel}" --with boto3 \
  python .github/exa-release/test-wheel-s3.py
```

This writes and repeatedly opens a LocalStack dataset, then requires repeated
HTTPS S3 requests to reach an AWS service response instead of failing during
DNS, TCP, or TLS setup.

## Publish

Publish each wheel with its `.provenance.txt`, and publish only the
`wheel_sha256` recorded in that provenance file; release assets are immutable.

## Optional reproducibility audit

`reproduce-wheel.sh` is not required for a release. It runs `build-wheel.sh`
twice from two sequential clean checkouts at the same canonical build path
(Rust ThinLTO output is sensitive to absolute build paths beyond diagnostic
path remapping) and fails unless both wheels have the same SHA-256. Run it
occasionally, for example after changing the build scripts or toolchain pins,
to confirm that builds remain bit-for-bit reproducible:

```shell
.github/exa-release/reproduce-wheel.sh x86_64 /persistent/path/exa-wheel-audit-x86_64
.github/exa-release/reproduce-wheel.sh aarch64 /persistent/path/exa-wheel-audit-aarch64
```

The verified wheel and provenance file are copied to `<release-dir>/verified`.
