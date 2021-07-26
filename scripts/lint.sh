python3 -m black vist tools
python3 -m isort vist tools
python3 -m pylint vist tools/*
python3 -m pydocstyle --convention=google vist tools
python3 -m mypy --strict vist tools
