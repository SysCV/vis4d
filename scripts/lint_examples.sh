python3 -m pylint examples --disable=duplicate-code,fixme,import-error
python3 -m pydocstyle --convention=google examples
python3 -m mypy --strict examples
python3 -m black examples
python3 -m isort examples
