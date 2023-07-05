# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


import os
import sys

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")

sys.path.insert(0, os.path.abspath(PATH_ROOT))


# -- Project information -----------------------------------------------------

project = "Vis4D"
copyright = "2022, ETH Zurich"
author = "Tobias Fischer"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
    "sphinx_design",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst", ".ipynb"]

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
pygments_dark_style = "monokai"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# TODO revise logo
html_theme_options = {
    "light_logo": "vis4d_logo.svg",
    "dark_logo": "vis4d_logo.svg",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

# html_favicon = '_static/img/logo_favicon.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

texinfo_documents = [
    (
        master_doc,
        "vis4d",
        "Vis4D Documentation",
        author,
        "Vis4D",
        "Dynamic Scene Understanding in Pytorch.",
        "Miscellaneous",
    )
]

# -- auto doc settings -------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "groupwise"
autoclass_content = "both"
add_module_names = False  # Remove namespaces from class/method signatures
autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl,__init__",
}

# -- Napoleon settings -------------------------------------------------------

# Settings for parsing non-sphinx style docstrings. We use Google style in this
# project.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- MYSTNB -----------------------------------------------------------------

suppress_warnings = ["mystnb.unknown_mime_type", "myst.header"]
nb_execution_mode = "off"
