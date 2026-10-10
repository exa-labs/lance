#!/usr/bin/env bash
# Validate prebuilt wheels on native hosts before publishing the immutable release.
#//<><gen-workflow><>#
# name: exa-release-v9.0.1-exa.19
# workflow_name: exa-release-v9.0.1-exa.19
# on:
#   push:
#     branches: [exa-release/v9.0.1-exa.19]
# permissions: {contents: read}
# concurrency:
#   group: exa-release-v9.0.1-exa.19
#   cancel-in-progress: false
# jobs:
#   smoke:
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
#       - uses: astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e
#       - name: Native LocalStack and HTTPS smoke test
#         run: bash release-tools/ci/exa-release.sh smoke ${{ matrix.arch }}
#   publish:
#     needs: smoke
#     runs-on: ubuntu-22.04
#     timeout-minutes: 15
#     permissions: {contents: write}
#     steps:
#       - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
#         with:
#           path: release-tools
#           persist-credentials: false
#       - name: Verify and publish immutable wheels
#         env:
#           GH_TOKEN: ${{ github.token }}
#         run: bash release-tools/ci/exa-release.sh publish
#//<>/<gen-workflow><>#
set -euo pipefail
source_commit=1aa82f9e26f7d75d77cbe69023d3cfa707615223
tag=v9.0.1-exa.19
version=9.0.1+exa.19
source_dir="$PWD/release-tools"
wheel_dir="$source_dir/release-wheels"
x86_sha256=bd8fe5716d9c8e17078481ad7739b921243a597d7655974ef08326ebfaa9e09e
arm_sha256=ffdc786c776bd3ef34a91430020c031b99474fea8db5d38c1ebdfc64c50d0a1e

check_provenance() {
  local wheel="${1}"
  local provenance="${wheel}.provenance.txt"
  unzip -p "${wheel}" '*/METADATA' | grep -Fx "Version: ${version}"
  grep -Fx "source_commit=${source_commit}" "${provenance}"
  local expected
  case "${wheel}" in
    *x86_64.whl) expected="${x86_sha256}" ;;
    *aarch64.whl) expected="${arm_sha256}" ;;
    *) echo "Unexpected wheel: ${wheel}" >&2; exit 1 ;;
  esac
  test "$(sha256sum "${wheel}" | cut -d ' ' -f1)" = "${expected}"
  grep -Fx "wheel_sha256=${expected}" "${provenance}"
  grep -Fx 'release: 1.91.0' "${provenance}"
  grep -Fx 'zig_version=0.16.0' "${provenance}"
  grep -Fx 'maturin_version=maturin 1.15.0' "${provenance}"
}

case "${1:?smoke|publish}" in
  smoke)
    arch="${2:?architecture}"
    test "$(uname -m)" = "${arch}"
    wheels=("${wheel_dir}/"*"${arch}.whl")
    test "${#wheels[@]}" = 1
    wheel="${wheels[0]}"
    check_provenance "${wheel}"
    bash "${source_dir}/.github/exa-release/validate-wheel.sh" "${wheel}" "${arch}"
    uv run --isolated --python 3.10 --with "${wheel}" --with boto3       python "${source_dir}/.github/exa-release/test-wheel-s3.py"
    ;;
  publish)
    wheels=("${wheel_dir}/"*.whl)
    test "${#wheels[@]}" = 2
    for arch in x86_64 aarch64; do
      wheel="${wheel_dir}/pylance-${version}-cp310-abi3-manylinux_2_17_${arch}.manylinux2014_${arch}.whl"
      bash "${source_dir}/.github/exa-release/validate-wheel.sh" "${wheel}" "${arch}"
      check_provenance "${wheel}"
    done
    gh release create "${tag}" --repo "${GITHUB_REPOSITORY}" --target "${source_commit}"       --title "v${version}" --notes "Release based on ${source_commit}, including #64 and #65.

Both manylinux2014 wheels use the unchanged pinned Rust 1.91.0, Zig 0.16.0, maturin 1.15.0 and protoc 24.4 build recipe. Both passed ELF/toolchain validation and native LocalStack and HTTPS S3 smoke tests. SHA-256 hashes and build inputs are recorded in the attached provenance files.

Multipart uploads call complete exactly once; throttled complete requests are retried at the HTTP layer."       "${wheel_dir}/"*.whl "${wheel_dir}/"*.provenance.txt
    ;;
  *) echo 'Usage: exa-release.sh smoke|publish [architecture]' >&2; exit 2 ;;
esac
