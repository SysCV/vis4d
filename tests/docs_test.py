"""Documentation tutorial notebook testing."""
import os

from pytest_notebook.nb_regression import NBRegressionFixture


def test_docs_tutorials() -> None:
    """Test tutorial notebooks."""
    ignores = (
        "/cells/*/metadata",
        "/cells/*/execution_count",
        "/cells/*/outputs/*/data/image",
        "/metadata/language_info/version",
    )
    replace = (
        ("/cells/*/outputs", "\\[Open3D INFO\\] [^\\n]+ *\\n?", ""),
        (
            "/cells/*/outputs",
            "Jupyter environment detected. Enabling Open3D WebVisualizer. *\\n?",
            "",
        ),
    )

    fixture = NBRegressionFixture(
        exec_timeout=50, diff_ignore=ignores, diff_replace=replace
    )
    fixture.diff_color_words = False
    doc_nb_path = "docs/source/tutorials/"
    for file in os.listdir(doc_nb_path):
        if file.endswith(".ipynb"):
            path = os.path.join(doc_nb_path, file)
            fixture.check(path)
