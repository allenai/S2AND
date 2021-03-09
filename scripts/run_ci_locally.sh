# Script for running the CI locally before making a PR
pytest tests/
flake8 s2and
flake8 scripts/*.py
black s2and --check --line-length 120
black scripts/*.py --check --line-length 120
bash ./scripts/mypy.sh
pytest tests/ --cov s2and --cov-fail-under=40