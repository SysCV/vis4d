"""Documentation tutorial notebook testing."""

from pytest_notebook.nb_regression import NBRegressionFixture

# This test is disabled because there is an issue with the libEGL.so.1 on
# the CI server for now.
# def test_3d_vis() -> None:
#     """Test tutorial notebooks."""
#     ignores = (
#         "/cells/*/metadata",
#         "/cells/*/execution_count",
#         "/cells/*/outputs/*/data/image",
#         "/metadata/language_info/version",
#     )
#     replace = (
#         ("/cells/*/outputs", "\\[Open3D INFO\\] [^\\n]+ *\\n?", ""),
#         (
#             "/cells/*/outputs",
#             "Jupyter environment detected. Enabling Open3D WebVisualizer.
#              *\\n?",
#             "",
#         ),
#     )

#     fixture = NBRegressionFixture(
#         exec_timeout=50, diff_ignore=ignores, diff_replace=replace
#     )
#     fixture.diff_color_words = False
#     file = "docs/source/tutorials/3D_visualization.ipynb"
#     fixture.check(file)


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
        "/cells/1/outputs/",  # Suppress downloading checkpoint output
        "/metadata/widgets",
        "/cells/5/outputs/",  # Suppress downloading checkpoint output
        "/cells/*/execution_count",
        "/cells/*/outputs/*/data/image",
        "/metadata/language_info/version",
    )
    replace = (("/cells/*/outputs", "\\[Open3D INFO\\] [^\\n]+ *\\n?", ""),)

    fixture = (
        NBRegressionFixture(  # Higher timeout for downloading checkpoints
            exec_timeout=300, diff_ignore=ignores, diff_replace=replace
        )
    )
    fixture.diff_color_words = False
    file = "docs/source/tutorials/getting_started.ipynb"
    fixture.check(file)


def test_cli() -> None:
    """Test the cli notebook."""
    # Lets for now ignore the output of the training cells. Cuda/Python version
    # on the server are different and the training is slightly non-det.
    ignores = (
        "/cells/2/outputs/",
        "/cells/4/outputs/",
        "/cells/*/metadata",
        "/metadata/language_info/version",
    )
    fixture = (
        NBRegressionFixture(  # Higher timeout for downloading checkpoints
            exec_timeout=300, diff_ignore=ignores
        )
    )
    fixture.diff_color_words = False
    file = "docs/source/tutorials/cli_interface.ipynb"
    fixture.check(file)
