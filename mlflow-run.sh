#!/bin/bash

set -euo pipefail

cd "$(dirname $0)/../output/logs"
uv tool run mlflow server --backend-store-uri sqlite:///mlflow.sqlite
