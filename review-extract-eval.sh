#!/bin/bash

set -euo pipefail

uv tool run \
    streamlit run review-extract-eval.py $@
