#!/bin/bash

set -euo pipefail

# Move all output files to dedicated directories

ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))
OUTPUT_DIR="${ROOT_DIR}/export"
MODEL_NAME=gemini-2.5-pro-exp-03-25


function move_files {
    local id=$1

    echo "Moving pkna-${id}"

    DEST="${OUTPUT_DIR}/pkna-${id}/${MODEL_NAME}"
    if [ -d "${DEST}" ]; then
        echo "Directory pkna-${id} already exists, skipping."
        continue
    fi

    mkdir "${DEST}"
    mv "${DEST}/../"*.json "${DEST}/"
}

move_files "0-2"
move_files "0-3"
