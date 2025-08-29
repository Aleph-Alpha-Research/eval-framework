from importlib.metadata import version

__version__ = version("eval-framework")
del version

__all__ = ["__version__"]
