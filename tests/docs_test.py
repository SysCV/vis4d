"""Documentation tutorial notebook testing."""
import os

from pytest_notebook.nb_regression import NBRegressionFixture


def test_docs_tutorials() -> None:
    """Test tutorial notebooks."""
    ignores = (
        "/cells/*/metadata",
        "/cells/*/execution_count",
        "/cells/*/outputs/*/data/image",
    )
    fixture = NBRegressionFixture(exec_timeout=50, diff_ignore=ignores)
    fixture.diff_color_words = False
    doc_nb_path = "docs/source/tutorials/"
    for file in os.listdir(doc_nb_path):
        if file.endswith(".ipynb"):
            path = os.path.join(doc_nb_path, file)
            fixture.check(path)
