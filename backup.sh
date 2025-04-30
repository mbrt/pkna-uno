#!/bin/bash

set -euo pipefail

ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))

# Backup exporter
cd "${ROOT_DIR}/exporter"
mkdir -p "${ROOT_DIR}/backup/exporter"
git ls-files | xargs -I{} cp --parents {} "${ROOT_DIR}/backup/exporter"

# Backup results
cd "${ROOT_DIR}"
mkdir -p "backup/results"
rsync -avz --include="*/" --include="*.json" --exclude="*" \
    export/ "${ROOT_DIR}/backup/results/" --prune-empty-dirs --delete

# Save list of files
find export/ -type f | sort > "${ROOT_DIR}/backup/files.txt"

# Compress backup
rm -f pkna-llm-backup.tar*
tar czf "pkna-llm-backup.tar" "backup"
bzip2 --best "pkna-llm-backup.tar"
