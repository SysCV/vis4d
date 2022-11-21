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
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    #'sphinx_autodoc_defaultargs',
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
source_suffix = ".rst"

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

html_theme_options = {
    #'light_logo': 'img/logo_light.svg',  TODO add logo
    #'dark_logo': 'img/logo_dark.svg',
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-sidebar-background": "#34495E",
        "color-sidebar-background-border": "#34495E",
        "color-sidebar-caption-text": "white",
        "color-sidebar-link-text--top-level": "white",
        "color-sidebar-link-text": "white",
        "sidebar-caption-font-size": "normal",
        "color-sidebar-item-background--hover": "#85929E",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#1a1c1e",
        "color-sidebar-background-border": "#1a1c1e",
        "color-sidebar-caption-text": "white",
        "color-sidebar-link-text--top-level": "white",
    },
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


# This is the expected signature of the handler for this event, cf doc
def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    # exclude unit tests from api doc
    return name.endswith("_test") or "unittest" in name


# Automatically called by sphinx at startup
def setup(app):
    # Connect the autodoc-skip-member event from apidoc to the callback
    app.connect("autodoc-skip-member", autodoc_skip_member_handler)


# auto doc options
autosummary_generate = True
autodoc_member_order = "groupwise"
autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}
