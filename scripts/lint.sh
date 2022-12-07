python3 -m black vis4d
python3 -m isort vis4d
python3 -m pylint vis4d
python3 -m pydocstyle --convention=google vis4d
python3 -m mypy vis4d

python3 -m black tests
python3 -m isort tests
python3 -m pylint tests
