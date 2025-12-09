# -- Path setup --------------------------------------------------------------
import os
import sys
from pallets_sphinx_themes import ProjectLink

# Add project root and src/ to sys.path
ROOT = os.path.abspath("..")     # eval-framework/
SRC = os.path.join(ROOT, "src")  # eval-framework/src/
sys.path.insert(0, SRC)

# -- Project information -----------------------------------------------------
project = "eval-framework"
author = "Aleph Alpha Research"
copyright = "2025, Aleph Alpha Research"
version = "0.2.5"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",  # ← add this
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autodoc_typehints = "description"
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"

# Optional: customize dark/light appearance
html_theme_options = {
    "light_logo": "white_logo.png",
    "dark_logo": "black_logo.png",
}

html_title = f"Eval-Framework Documentation ({version})"
html_static_path = ["_static"]
html_show_sourcelink = False

html_context = {
    "project_links": [
        ProjectLink("PyPI", "https://pypi.org/project/eval-framework/"),
        ProjectLink("GitHub", "https://github.com/Aleph-Alpha-Research/eval-framework"),
    ]
}
