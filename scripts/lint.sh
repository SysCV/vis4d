python3 -m pylint systm
python3 -m flake8 --docstring-convention google systm
python3 -m mypy --strict systm
python3 -m black --check systm
python3 -m isort -c systm/**/*.py