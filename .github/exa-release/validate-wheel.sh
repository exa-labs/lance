#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <wheel> <x86_64|aarch64>" >&2
  exit 1
fi

wheel="$(realpath "$1")"
platform="$2"

case "${platform}" in
  x86_64)
    expected_machine="Advanced Micro Devices X86-64"
    expected_loader="ld-linux-x86-64.so.2"
    ;;
  aarch64)
    expected_machine="AArch64"
    expected_loader="ld-linux-aarch64.so.1"
    ;;
  *)
    echo "Unsupported platform: ${platform}" >&2
    exit 1
    ;;
esac

test -f "${wheel}"
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

wheel_metadata="$(unzip -p "${wheel}" '*/WHEEL')"
grep -F "Generator: maturin (1.15.0)" <<<"${wheel_metadata}"
grep -F "manylinux_2_17_${platform}" <<<"${wheel_metadata}"

shared_object="$(unzip -Z1 "${wheel}" | sed -n '/lance\/lance\.abi3\.so$/p')"
test -n "${shared_object}"
unzip -p "${wheel}" "${shared_object}" > "${work_dir}/lance.abi3.so"

machine="$(readelf -h "${work_dir}/lance.abi3.so" | sed -n 's/^[[:space:]]*Machine:[[:space:]]*//p')"
test "${machine}" = "${expected_machine}"

comments="$(readelf -p .comment "${work_dir}/lance.abi3.so")"
grep -F "clang version 21.1.0" <<<"${comments}"
grep -F "rustc version 1.91.0" <<<"${comments}"
grep -F "Linker: LLD 21.1.0" <<<"${comments}"

dependencies="$(readelf -d "${work_dir}/lance.abi3.so")"
needed="$(
  sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' <<<"${dependencies}" |
    sort
)"
expected_needed="$(
  printf '%s\n' \
    "${expected_loader}" \
    "libc.so.6" \
    "libdl.so.2" \
    "libm.so.6" \
    "libpthread.so.0" |
    sort
)"
if [[ "${needed}" != "${expected_needed}" ]]; then
  echo "ELF dependencies do not match exa.9:" >&2
  diff -u <(printf '%s\n' "${expected_needed}") <(printf '%s\n' "${needed}") >&2
  exit 1
fi

max_glibc="$(
  readelf --version-info "${work_dir}/lance.abi3.so" |
    sed -n 's/.*Name: GLIBC_\([0-9.]*\).*/\1/p' |
    sort -V |
    tail -1
)"
test "${max_glibc}" = "2.17"
sha256sum "${wheel}"
