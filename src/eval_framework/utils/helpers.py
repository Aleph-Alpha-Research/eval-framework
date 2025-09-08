def count_bytes(text: str, /, *, encoding: str = "utf-8") -> int:
    """Count the number of bytes in a string."""
    return len(bytes(text.encode(encoding)))
