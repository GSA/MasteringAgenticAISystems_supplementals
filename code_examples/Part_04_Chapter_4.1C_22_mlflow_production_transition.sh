#!/bin/bash
# Transition MLflow model to production stage

# Transition to production stage
mlflow models transition --model-name customer-support-agent \
                        --version 1.3.0 \
                        --stage Production \
                        --archive-existing-versions
