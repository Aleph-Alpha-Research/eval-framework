import logging
from pathlib import Path

VERBOSITY_MAP = {
    0: logging.CRITICAL,
    1: logging.INFO,
    2: logging.DEBUG,
}


def setup_logging(
    output_dir: Path | None = None, log_level: int = 1, log_filename: str = "evaluation.log"
) -> logging.Logger:
    """
    Set up centralized logging configuration for the entire framework.

    Args:
        output_dir: Directory to save log files. If None, no file handler is attached.
        log_level: Logging level (default: INFO)
        log_filename: Name of the log file

    Returns:
        Configured root logger
    """
    # Map verbosity integer to logging level
    mapped_log_level = VERBOSITY_MAP.get(log_level, logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(mapped_log_level)

    # File handler (if output directory provided). No console handler: keeps stdout free for tqdm.
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / log_filename

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(mapped_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging configured. File: {log_file}")

    return root_logger
