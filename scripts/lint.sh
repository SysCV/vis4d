python3 -m black vis4d projects
python3 -m isort vis4d projects
python3 -m pylint vis4d projects
python3 -m pydocstyle --convention=google vis4d projects
python3 -m mypy --strict vis4d
python3 -m mypy projects
