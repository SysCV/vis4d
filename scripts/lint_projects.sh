python3 -m black projects
python3 -m isort projects
python3 -m pylint projects
python3 -m pydocstyle --convention=google projects
python3 -m mypy --strict projects
