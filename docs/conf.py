# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pallets_sphinx_themes import ProjectLink
# point to your package; adjust path if package folder at repo root
sys.path.insert(0, os.path.abspath('../src/eval_framework/'))

project = 'eval-framework'
copyright = '2025, Aleph Alpha Research'
author = 'Aleph Alpha Research'
version = '0.2.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',          # generate docs from docstrings
    'sphinx.ext.napoleon',         # Google / NumPy style docstrings
    'sphinx.ext.autosummary',      # ← built-in, no pip install needed
    'sphinx.ext.viewcode',         # link to highlighted source
    'sphinx_autodoc_typehints',    # show type hints
    "pallets_sphinx_themes",       # Flask theme
]

autosummary_generate = True  # turn on sphinx.ext.autosummary
autodoc_typehints = "description"  # put typehints into descriptions

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "flask"
html_theme_options = {"index_sidebar_logo": False}
html_context = {
    "project_links": [
        ProjectLink("PyPI Releases", "https://pypi.org/project/eval-framework/"),
        ProjectLink("Source Code", "https://github.com/Aleph-Alpha-Research/eval-framework"),
    ]
}
html_sidebars = {
    "**": ["project.html", "localtoc.html", "relations.html", "searchbox.html"]
}
singlehtml_sidebars = {
    "**": ["project.html", "localtoc.html"]
}
html_static_path = ["_static"]
html_title = f"Eval-Framework Documentation ({version})"
html_show_sourcelink = False
