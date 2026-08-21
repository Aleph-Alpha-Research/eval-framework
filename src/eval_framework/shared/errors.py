import os


def raise_errors() -> bool:
    debug = os.environ.get("DEBUG", "FALSE").lower()
    if debug in {"1", "true"}:
        return True
    elif debug in {"0", "false"}:
        return False
    else:
        raise ValueError(f"Invalid value for DEBUG environment variable: {debug}. Use one of 1, 0, true, false.")
