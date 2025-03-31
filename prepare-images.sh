#!/bin/bash

set -euo pipefail

ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))
EXPORT_DIR="${ROOT_DIR}/tmp"

FROM_NUM=9
TO_NUM=10

for i in $(seq $FROM_NUM $TO_NUM); do
    echo "Processing PNKA-$i"
    mkdir -p "${EXPORT_DIR}/pkna-$i"
    cd "${EXPORT_DIR}/pkna-$i"
    unzip "${ROOT_DIR}/orig/PKNA $i.cbz"
done
