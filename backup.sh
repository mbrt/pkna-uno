#!/bin/bash

set -exuo pipefail

ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))

# Backup exporter
cd "${ROOT_DIR}/exporter"
mkdir -p "${ROOT_DIR}/backup/exporter"
git ls-files | xargs -I{} cp --parents {} "${ROOT_DIR}/backup/exporter"

# Backup results
cd "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/backup/results"
cp export/characters.json "${ROOT_DIR}/backup/results"
for d in export/pkna-*; do
    mkdir -p "${ROOT_DIR}/backup/results/$(basename "$d")"
    cp "${d}/"*.json "${ROOT_DIR}/backup/results/$(basename "
$d")" || true
done

# Compress backup
cd "${ROOT_DIR}"
rm -f pkna-llm-backup.tar*
tar czf "pkna-llm-backup.tar" "backup"
bzip2 --best "pkna-llm-backup.tar"
