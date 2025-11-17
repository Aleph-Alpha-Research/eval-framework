import logging
import sys
from pathlib import Path
from tqdm import tqdm

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
        output_dir: Directory to save log files. If None, logs only to console.
        log_level: Logging level (default: INFO)
        log_filename: Name of the log file

    Returns:
        Configured root logger
    """
    # Map verbosity integer to logging level
    log_level = VERBOSITY_MAP.get(log_level, logging.INFO)
    
    # Basic configuration
    logging.basicConfig(level=log_level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if output directory provided)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / log_filename

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging configured. File: {log_file}")
    else:
        root_logger.info("Logging configured (console only)")

    root_logger.info(f"Output directory for logs: {output_dir if output_dir else 'None'}")

    return root_logger