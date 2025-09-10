#!/usr/bin/env bash

# Make sure that pytorch is not a main dependency of the project
if uv export --locked --no-dev --no-hashes --no-annotate --format='requirements.txt' | grep -i torch; then
    echo "Error: PyTorch is a main dependency of the project, which should not be the case."
    exit 1
fi
