# Script for running the CI locally before making a PR
pytest tests/
black s2and --check --line-length 120
black scripts/*.py --check --line-length 120
bash ./scripts/mypy.sh
pytest tests/ --cov s2and --cov-fail-under=40