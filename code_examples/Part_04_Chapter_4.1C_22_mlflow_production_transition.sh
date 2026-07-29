#!/usr/bin/env bash
# Assign a stable deployment alias to a registered MLflow model version.
# Model registry stages are deprecated; aliases provide a named reference.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-customer-support-agent}"
MODEL_VERSION="${MODEL_VERSION:-1}"
MODEL_ALIAS="${MODEL_ALIAS:-champion}"

python3 - "$MODEL_NAME" "$MODEL_VERSION" "$MODEL_ALIAS" <<'PY'
from mlflow import MlflowClient
import sys

model_name, model_version, model_alias = sys.argv[1:]
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias=model_alias,
    version=model_version,
)
print(f"Assigned alias {model_alias!r} to {model_name!r} version {model_version}")
PY
