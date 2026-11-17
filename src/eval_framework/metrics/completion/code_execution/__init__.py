"""Shared infrastructure for metrics that score code by executing it.

- ``sandbox``: running code in Docker containers (Python snippets, or any language in a language-specific image).
- ``python_tests``: checking Python code against a test suite (merging, dependency installs, output parsing).
"""
