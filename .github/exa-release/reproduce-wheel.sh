#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <x86_64|aarch64> <release-dir>" >&2
  exit 1
fi

platform="$1"
release_dir="$2"
repo_root="$(git rev-parse --show-toplevel)"
source_commit="$(git rev-parse HEAD)"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Release source checkout must not contain tracked changes" >&2
  exit 1
fi

mkdir -p "${release_dir}"
release_dir="$(realpath "${release_dir}")"
if [[ -n "$(find "${release_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Release directory must be empty: ${release_dir}" >&2
  exit 1
fi

work_dir="${release_dir}/work"
source_dir="${work_dir}/source"
output_dir="${work_dir}/dist"

for run in 1 2; do
  run_dir="${release_dir}/run-${run}"
  rm -rf -- "${work_dir}"
  git clone --no-hardlinks --no-checkout "${repo_root}" "${source_dir}"
  git -C "${source_dir}" checkout --detach "${source_commit}"
  "${source_dir}/.github/exa-release/build-wheel.sh" \
    "${source_dir}" \
    "${platform}" \
    "${output_dir}"
  rm -rf -- "${output_dir}/target"
  mkdir -p "${run_dir}"
  cp "${output_dir}"/*.whl "${output_dir}"/*.provenance.txt "${run_dir}/"
done
rm -rf -- "${work_dir}"

wheel_one="$(find "${release_dir}/run-1" -maxdepth 1 -name "*${platform}*.whl" -print -quit)"
wheel_two="$(find "${release_dir}/run-2" -maxdepth 1 -name "*${platform}*.whl" -print -quit)"
test -n "${wheel_one}"
test -n "${wheel_two}"

sha_one="$(sha256sum "${wheel_one}" | cut -d ' ' -f1)"
sha_two="$(sha256sum "${wheel_two}" | cut -d ' ' -f1)"
if [[ "${sha_one}" != "${sha_two}" ]]; then
  echo "Clean builds produced different wheel hashes:" >&2
  echo "run-1: ${sha_one}" >&2
  echo "run-2: ${sha_two}" >&2
  exit 1
fi

cmp "${wheel_one}.provenance.txt" "${wheel_two}.provenance.txt"
mkdir -p "${release_dir}/verified"
cp "${wheel_one}" "${release_dir}/verified/"
cp "${wheel_one}.provenance.txt" "${release_dir}/verified/"
echo "Reproduced ${platform} wheel: ${sha_one}"
