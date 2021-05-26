python3 -m black openmt tools
python3 -m isort openmt tools
python3 -m pylint openmt tools
python3 -m pydocstyle --convention=google openmt tools
python3 -m mypy --strict openmt tools