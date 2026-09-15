#!/usr/bin/env bash
set -euo pipefail

wheel="$(find dist -maxdepth 1 -name "*${PLATFORM}*.whl" -print -quit)"
test -n "${wheel}"

wheel_metadata="$(unzip -p "${wheel}" '*/WHEEL')"
grep -F "Generator: maturin (1.15.0)" <<<"${wheel_metadata}"
grep -F "manylinux_2_17_${PLATFORM}" <<<"${wheel_metadata}"

shared_object="$(unzip -Z1 "${wheel}" | sed -n '/lance\/lance\.abi3\.so$/p')"
test -n "${shared_object}"
unzip -p "${wheel}" "${shared_object}" > lance.abi3.so

comments="$(readelf -p .comment lance.abi3.so)"
grep -F "clang version 21.1.0" <<<"${comments}"
grep -F "rustc version 1.91.0" <<<"${comments}"
grep -F "Linker: LLD 21.1.0" <<<"${comments}"

dependencies="$(readelf -d lance.abi3.so)"
if grep -F "libgcc_s.so.1" <<<"${dependencies}"; then
  echo "Unexpected libgcc_s dependency" >&2
  exit 1
fi

max_glibc="$(
  readelf --version-info lance.abi3.so |
    sed -n 's/.*Name: GLIBC_\([0-9.]*\).*/\1/p' |
    sort -V |
    tail -1
)"
test "${max_glibc}" = "2.17"
sha256sum "${wheel}"
