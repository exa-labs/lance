#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 3 ]]; then
  echo "Usage: $0 <source-dir> <x86_64|aarch64> <output-dir>" >&2
  exit 1
fi

source_dir="$(realpath "$1")"
platform="$2"
output_dir="$3"

case "${platform}" in
  x86_64)
    target="x86_64-unknown-linux-gnu"
    ;;
  aarch64)
    target="aarch64-unknown-linux-gnu"
    ;;
  *)
    echo "Unsupported platform: ${platform}" >&2
    exit 1
    ;;
esac

test "$(uname -s)" = "Linux"
test "$(uname -m)" = "x86_64"
test -x /usr/bin/clang
test -x /usr/bin/clang++

unset ARCHFLAGS
unset _PYTHON_HOST_PLATFORM
unset _PYTHON_SYSCONFIGDATA_NAME

for command in curl diff git pkg-config readelf rustup sha256sum tar unzip xz; do
  command -v "${command}" >/dev/null
done

if ! git -C "${source_dir}" diff --quiet ||
  ! git -C "${source_dir}" diff --cached --quiet; then
  echo "Source checkout must not contain tracked changes" >&2
  exit 1
fi

mkdir -p "${output_dir}"
output_dir="$(realpath "${output_dir}")"
if [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Output directory must be empty: ${output_dir}" >&2
  exit 1
fi

toolchain_dir="${EXA_WHEEL_TOOLCHAIN_DIR:-${HOME}/.cache/lance-exa-wheel-toolchain}"
mkdir -p "${toolchain_dir}"

download() {
  local url="$1"
  local expected_sha256="$2"
  local destination="$3"

  if [[ ! -f "${destination}" ]]; then
    curl --fail --location --retry 5 --output "${destination}" "${url}"
  fi
  echo "${expected_sha256}  ${destination}" | sha256sum --check
}

zig_archive="${toolchain_dir}/zig-x86_64-linux-0.16.0.tar.xz"
zig_dir="${toolchain_dir}/zig-0.16.0"
download \
  "https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz" \
  "70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00" \
  "${zig_archive}"
if [[ ! -x "${zig_dir}/zig" ]]; then
  mkdir -p "${zig_dir}"
  tar -xJf "${zig_archive}" -C "${zig_dir}" --strip-components=1
fi
test "$("${zig_dir}/zig" version)" = "0.16.0"

maturin_archive="${toolchain_dir}/maturin-x86_64-unknown-linux-musl-1.15.0.tar.gz"
maturin_dir="${toolchain_dir}/maturin-1.15.0"
download \
  "https://github.com/PyO3/maturin/releases/download/v1.15.0/maturin-x86_64-unknown-linux-musl.tar.gz" \
  "5a7d5226e83598b0c85881a0a03be16393ee2c0924b6f9d752951795825ac806" \
  "${maturin_archive}"
if [[ ! -x "${maturin_dir}/maturin" ]]; then
  mkdir -p "${maturin_dir}"
  tar -xzf "${maturin_archive}" -C "${maturin_dir}"
fi
test "$("${maturin_dir}/maturin" --version)" = "maturin 1.15.0"

protoc_archive="${toolchain_dir}/protoc-24.4-linux-x86_64.zip"
protoc_dir="${toolchain_dir}/protoc-24.4"
download \
  "https://github.com/protocolbuffers/protobuf/releases/download/v24.4/protoc-24.4-linux-x86_64.zip" \
  "5871398dfd6ac954a6adebf41f1ae3a4de915a36a6ab2fd3e8f2c00d45b50dec" \
  "${protoc_archive}"
if [[ ! -x "${protoc_dir}/bin/protoc" ]]; then
  mkdir -p "${protoc_dir}"
  unzip -q "${protoc_archive}" -d "${protoc_dir}"
fi
test "$("${protoc_dir}/bin/protoc" --version)" = "libprotoc 24.4"

rustup toolchain install 1.91.0 --profile minimal --target "${target}"
rust_version="$(rustup run 1.91.0 rustc --version --verbose)"
grep -F "release: 1.91.0" <<<"${rust_version}"
grep -F "commit-hash: f8297e351a40c1439a467bbbb6879088047f50b3" <<<"${rust_version}"

source_commit="$(git -C "${source_dir}" rev-parse HEAD)"
source_date_epoch="$(git -C "${source_dir}" show -s --format=%ct HEAD)"
target_dir="${output_dir}/target"
path_flags="-ffile-prefix-map=${source_dir}=/workspace/lance -fdebug-prefix-map=${source_dir}=/workspace/lance"
rust_flags="--remap-path-prefix=${source_dir}=/workspace/lance --remap-path-prefix=${target_dir}=/workspace/target"
target_suffix="${target//-/_}"
target_cflags="${path_flags}"
aws_lc_cflags="${path_flags}"

if [[ "${platform}" = "x86_64" ]]; then
  rust_flags="-C target-cpu=haswell -C target-feature=+avx2,+fma,+f16c ${rust_flags}"
else
  target_cflags="-D__ARM_ARCH=8 ${path_flags}"
  aws_lc_cflags="-DAT_HWCAP2=26 ${path_flags}"
fi

target_env="${target^^}"
target_env="${target_env//-/_}"
export "CARGO_TARGET_${target_env}_RUSTFLAGS=${rust_flags}"
export "AWS_LC_SYS_CFLAGS_${target_suffix}=${aws_lc_cflags}"
export "CFLAGS_${target_suffix}=${target_cflags}"
export "CXXFLAGS_${target_suffix}=${target_cflags}"
export CC=/usr/bin/clang
export CARGO_INCREMENTAL=0
export CARGO_NET_GIT_FETCH_WITH_CLI=true
export CARGO_TARGET_DIR="${target_dir}"
export CXX=/usr/bin/clang++
export LANG=C
export LC_ALL=C
export PATH="${zig_dir}:${protoc_dir}/bin:${PATH}"
export PYTHONHASHSEED=0
export RUSTUP_TOOLCHAIN=1.91.0
export SOURCE_DATE_EPOCH="${source_date_epoch}"
export TZ=UTC
export ZERO_AR_DATE=1

if [[ "${platform}" = "aarch64" ]]; then
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/clang
fi

(
  cd "${source_dir}/python"
  "${maturin_dir}/maturin" build \
    --release \
    --strip \
    --locked \
    --zig \
    --target "${target}" \
    --compatibility manylinux2014 \
    --out "${output_dir}"
)

wheel="$(find "${output_dir}" -maxdepth 1 -name "*${platform}*.whl" -print -quit)"
test -n "${wheel}"
"${source_dir}/.github/exa-release/validate-wheel.sh" "${wheel}" "${platform}"

{
  echo "source_commit=${source_commit}"
  echo "source_date_epoch=${source_date_epoch}"
  echo "target=${target}"
  echo "zig_archive_sha256=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00"
  echo "maturin_archive_sha256=5a7d5226e83598b0c85881a0a03be16393ee2c0924b6f9d752951795825ac806"
  echo "protoc_archive_sha256=5871398dfd6ac954a6adebf41f1ae3a4de915a36a6ab2fd3e8f2c00d45b50dec"
  echo "${rust_version}"
  "${zig_dir}/zig" env
  "${maturin_dir}/maturin" --version
  sha256sum "${wheel}"
} > "${wheel}.provenance.txt"
