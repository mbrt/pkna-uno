#!/bin/bash

set -euo pipefail

ROOT_DIR=$(dirname "$(realpath "$0")")
cd "${ROOT_DIR}"

# Start fresh
rm -f pkna-llm-backup.tar

# Backup exporter files
git ls-files | xargs tar cf pkna-llm-backup.tar

# Save list of files in input
find input/ -type f | sort > input-files.txt

# Backup some input files and output files
tar --append -f pkna-llm-backup.tar \
    --exclude='input/characters' \
    --exclude='input/models' \
    --exclude='input/orig' \
    --exclude='input/pkna' \
    --exclude='input/schede' \
    --exclude='**/mlartifacts' \
    --exclude='output/sft/smoke_test' \
    input/ output/ input-files.txt

# Compress backup
rm -f pkna-llm-backup.tar.*
xz "pkna-llm-backup.tar"
