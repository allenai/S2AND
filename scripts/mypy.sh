#!/usr/bin/env bash
# Run type checking over the python code.

mypy s2and --ignore-missing-imports
mypy scripts/*.py --ignore-missing-imports