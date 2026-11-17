"""Metrics that score code by executing it, and the infrastructure they share.

Infrastructure:

- ``sandbox``: running code in Docker containers (Python snippets, or any language in a language-specific image).
- ``python_tests``: checking Python code against a test suite (merging, dependency installs, output parsing).

Metrics:

- ``python_assertion``: Python code whose last printed token must be ``True`` (HumanEval, MBPP, LBPP).
- ``pass_at_one``: pass@1 of Python code against a unittest suite (BigCodeBench).
- ``multipl_e_assertion``: MultiPL-E's language-native test harnesses, in a language-specific image.
"""
