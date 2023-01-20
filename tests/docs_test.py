"""Documentation tutorial notebook testing."""

from pytest_notebook.nb_regression import NBRegressionFixture


def test_3d_vis() -> None:
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
    file = "docs/source/tutorials/3D_visualization.ipynb"
    fixture.check(file)


def test_vis() -> None:
    """Test visualization notebooks."""
    ignores = (
        "/cells/*/metadata",
        "/cells/*/execution_count",
        "/cells/*/outputs/*/data/image",
        "/cells/1/outputs/0",
        "/metadata/language_info/version",
    )
    replace = (
        ("/cells/*/outputs", "\\[Open3D INFO\\] [^\\n]+ *\\n?", ""),
        (
            "/cells/*/outputs",
            "Jupyter environment detected. Enabling Open3D WebVisualizer. *\\n?",  # pylint: disable=line-too-long
            "",
        ),
    )

    fixture = NBRegressionFixture(
        exec_timeout=50, diff_ignore=ignores, diff_replace=replace
    )
    fixture.diff_color_words = False
    file = "docs/source/tutorials/visualization.ipynb"
    fixture.check(file)


def test_get_started() -> None:
    """Test get started notebooks."""
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
            "Jupyter environment detected. Enabling Open3D WebVisualizer. *\\n?",  # pylint: disable=line-too-long
            "",
        ),
    )

    fixture = NBRegressionFixture(
        exec_timeout=50, diff_ignore=ignores, diff_replace=replace
    )
    fixture.diff_color_words = False
    file = "docs/source/tutorials/getting_started.ipynb"
    fixture.check(file)
