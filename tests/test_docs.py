"""Documentation tutorial notebook testing."""
import os

from pytest_notebook.nb_regression import NBRegressionFixture


def test_docs_tutorials():
    """Test tutorial notebooks."""
    fixture = NBRegressionFixture(exec_timeout=50)
    fixture.diff_color_words = False
    doc_nb_path = "docs/source/tutorials/"
    for file in os.listdir(doc_nb_path):
        if file.endswith(".ipynb"):
            path = os.path.join(doc_nb_path, file)
            fixture.check(path)
