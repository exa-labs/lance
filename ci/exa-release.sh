#!/usr/bin/env bash
# Build and validate both wheels before creating the immutable release.
#//<><gen-workflow><>#
# name: exa-release-v9.0.1-exa.18
# workflow_name: exa-release-v9.0.1-exa.18
# on:
#   push:
#     branches: [exa-release/v9.0.1-exa.18]
# permissions: {contents: read}
# concurrency:
#   group: exa-release-v9.0.1-exa.18
#   cancel-in-progress: false
# jobs:
#   build:
#     runs-on: ubuntu-22.04
#     timeout-minutes: 180
#     strategy:
#       fail-fast: false
#       matrix:
#         arch: [x86_64, aarch64]
#     steps:
#       - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
#         with:
#           path: release-tools
#           persist-credentials: false
#       - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
#         with:
#           ref: 3b75dabdd910d028864406c893e2fb7584758c36
#           path: source
#           persist-credentials: false
#       - name: Install host tools
#         run: sudo apt-get update && sudo apt-get install -y clang pkg-config unzip xz-utils
#       - name: Build with pinned release toolchain
#         env:
#           CARGO_BUILD_JOBS: '2'
#         run: bash release-tools/ci/exa-release.sh build ${{ matrix.arch }}
#       - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02
#         with:
#           name: wheel-${{ matrix.arch }}
#           path: |
#             dist/${{ matrix.arch }}/*.whl
#             dist/${{ matrix.arch }}/*.provenance.txt
#           if-no-files-found: error
#   smoke:
#     needs: build
#     timeout-minutes: 30
#     strategy:
#       fail-fast: false
#       matrix:
#         include:
#           - arch: x86_64
#             runner: ubuntu-22.04
#           - arch: aarch64
#             runner: ubuntu-24.04-arm
#     runs-on: ${{ matrix.runner }}
#     services:
#       localstack:
#         image: localstack/localstack:4.0
#         env:
#           SERVICES: s3,dynamodb,kms
#           AWS_ACCESS_KEY_ID: ACCESS_KEY
#           AWS_SECRET_ACCESS_KEY: SECRET_KEY
#         ports: ['4566:4566']
#         options: --health-cmd "curl -fsS http://localhost:4566/_localstack/health" --health-interval 5s --health-timeout 5s --health-retries 20
#     steps:
#       - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
#         with:
#           path: release-tools
#           persist-credentials: false
#       - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
#         with:
#           ref: 3b75dabdd910d028864406c893e2fb7584758c36
#           path: source
#           persist-credentials: false
#       - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093
#         with:
#           name: wheel-${{ matrix.arch }}
#           path: dist/${{ matrix.arch }}
#       - uses: astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e
#       - name: Native LocalStack and HTTPS smoke test
#         run: bash release-tools/ci/exa-release.sh smoke ${{ matrix.arch }}
#   publish:
#     needs: [build, smoke]
#     runs-on: ubuntu-22.04
#     timeout-minutes: 15
#     permissions: {contents: write}
#     steps:
#       - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
#         with:
#           path: release-tools
#           persist-credentials: false
#       - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
#         with:
#           ref: 3b75dabdd910d028864406c893e2fb7584758c36
#           path: source
#           persist-credentials: false
#       - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093
#         with:
#           pattern: wheel-*
#           path: dist
#           merge-multiple: true
#       - name: Verify and publish immutable wheels
#         env:
#           GH_TOKEN: ${{ github.token }}
#         run: bash release-tools/ci/exa-release.sh publish
#//<>/<gen-workflow><>#
set -euo pipefail
source_commit=3b75dabdd910d028864406c893e2fb7584758c36
tag=v9.0.1-exa.18
version=9.0.1+exa.18
source_dir="$PWD/source"

check_provenance() {
  local wheel="${1}"
  local provenance="${wheel}.provenance.txt"
  grep -Fx "source_commit=${source_commit}" "${provenance}"
  grep -Fx "wheel_sha256=$(sha256sum "${wheel}" | cut -d ' ' -f1)" "${provenance}"
  grep -Fx 'release: 1.91.0' "${provenance}"
  grep -Fx 'zig_version=0.16.0' "${provenance}"
  grep -Fx 'maturin_version=maturin 1.15.0' "${provenance}"
}

case "${1:?build|smoke|publish}" in
  build)
    arch="${2:?architecture}"
    output="${RUNNER_TEMP}/exa-wheel-${arch}"
    bash "${source_dir}/.github/exa-release/build-wheel.sh" "${source_dir}" "${arch}" "${output}"
    mkdir -p "dist/${arch}"
    cp "${output}"/*.whl "${output}"/*.provenance.txt "dist/${arch}/"
    ;;
  smoke)
    arch="${2:?architecture}"
    test "$(uname -m)" = "${arch}"
    wheels=("$PWD/dist/${arch}/"*.whl)
    test "${#wheels[@]}" = 1
    wheel="${wheels[0]}"
    check_provenance "${wheel}"
    uv run --isolated --python 3.10 --with "${wheel}" --with boto3       python "${source_dir}/.github/exa-release/test-wheel-s3.py"
    ;;
  publish)
    wheels=(dist/*.whl)
    test "${#wheels[@]}" = 2
    for arch in x86_64 aarch64; do
      wheel="dist/pylance-${version}-cp310-abi3-manylinux_2_17_${arch}.manylinux2014_${arch}.whl"
      bash "${source_dir}/.github/exa-release/validate-wheel.sh" "${wheel}" "${arch}"
      check_provenance "${wheel}"
    done
    gh release create "${tag}" --repo "${GITHUB_REPOSITORY}" --target "${source_commit}"       --title "v${version}" --notes "Release based on ${source_commit}, including #58 and #60.

Both manylinux2014 wheels use the unchanged pinned Rust 1.91.0, Zig 0.16.0, maturin 1.15.0 and protoc 24.4 build recipe. Both passed ELF/toolchain validation and native LocalStack and HTTPS S3 smoke tests. SHA-256 hashes and build inputs are recorded in the attached provenance files.

Column chunking is opt-in through the Rust writer strategy; Python dataset writes retain the default strategy."       dist/*.whl dist/*.provenance.txt
    ;;
  *) echo 'Usage: ci-release.sh build|smoke|publish [architecture]' >&2; exit 2 ;;
esac
