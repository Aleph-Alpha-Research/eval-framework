# -- Path setup --------------------------------------------------------------
import os
import sys

# Minimal: we can skip adding project src if we're not testing autodoc
# sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "eval-framework"
author = "Aleph Alpha Research"
copyright = "2025, Aleph Alpha Research"
version = "0.2.5"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # Markdown support
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "alabaster"  # simple built-in theme for faster builds
html_static_path = ["_static"]
html_show_sourcelink = False
html_title = f"Eval-Framework Docs Test ({version})"
