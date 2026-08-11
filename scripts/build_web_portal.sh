#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_base_path="${DYNNAV_SITE_BASE_PATH:-}"
researcher_base_path="${site_base_path}/researcher"
output_dir="${repo_root}/.web-dist"

DYNNAV_SITE_BASE_PATH="${site_base_path}" npm --prefix "${repo_root}/website" run build
DYNNAV_RESEARCHER_BASE_PATH="${researcher_base_path}" npm --prefix "${repo_root}/apps/web" run build

rm -rf "${output_dir}"
mkdir -p "${output_dir}/researcher"
cp -a "${repo_root}/website/out/." "${output_dir}/"
cp -a "${repo_root}/apps/web/out/." "${output_dir}/researcher/"
touch "${output_dir}/.nojekyll"

printf 'Built DynNav portal at %s (site base path: %s)\n' "${output_dir}" "${site_base_path:-/}"
