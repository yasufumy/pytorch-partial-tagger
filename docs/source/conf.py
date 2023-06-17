# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pytorch-partial-tagger"
copyright = "2023, Yasufumi Taniguchi"
author = "Yasufumi Taniguchi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]
source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = ["_build", ".DS_Store"]

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = project

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/yasufumy/pytorch-partial-tagger",
    "use_repository_button": True,
}

html_static_path = ["_static"]

always_document_param_types = True
