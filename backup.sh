#!/bin/bash

set -euo pipefail

ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))

# Backup exporter
cd "${ROOT_DIR}/exporter"
rm -rf "${ROOT_DIR}/backup/exporter"
mkdir -p "${ROOT_DIR}/backup/exporter"
git ls-files | xargs -I{} cp --parents {} "${ROOT_DIR}/backup/exporter"

# Backup results
cd "${ROOT_DIR}"
mkdir -p "backup/output"
rsync -avz output/ "${ROOT_DIR}/backup/output/" --prune-empty-dirs --delete

# Backup some input files
mkdir -p "${ROOT_DIR}/backup/input"
rsync -avz input/ "${ROOT_DIR}/backup/input/" --prune-empty-dirs --delete --exclude 'pkna/' --exclude 'schede/' --exclude 'characters/'

# Save list of files in input
find input/ -type f | sort > "${ROOT_DIR}/backup/input-files.txt"

# Compress backup
rm -f pkna-llm-backup.tar*
tar czf "pkna-llm-backup.tar" "backup"
bzip2 --best "pkna-llm-backup.tar"
