import logging
from pathlib import Path
from eval_framework.utils.logging import setup_logging
from eval_framework.utils.tqdm_handler import safe_tqdm_write, get_disable_bar_flag, logger as tqdm_logger

def test_logging_level(tmp_path: Path) -> None:
    """
    Test that the logging setup correctly configures logging levels.
    """
    # Set up logging with different verbosity levels
    logger = setup_logging(output_dir=tmp_path, log_level=0)
    assert logger.level == logging.CRITICAL

    logger = setup_logging(output_dir=tmp_path, log_level=1)
    assert logger.level == logging.INFO

    logger = setup_logging(output_dir=tmp_path, log_level=2)
    assert logger.level == logging.DEBUG

def test_tqdm_logging(tmp_path: Path) -> None:
    """
    Test that tqdm logging integration works correctly.
    """
    # Set up logging
    setup_logging(output_dir=tmp_path, log_level=1)

    # Ensure safe_tqdm_write works without error
    safe_tqdm_write("This is a test message.", level=logging.INFO)

    # Test get_disable_bar_flag
    disable_flag = get_disable_bar_flag()
    assert disable_flag is False  # INFO level should not disable the bar
    assert tqdm_logger.getEffectiveLevel() == logging.INFO

    # Change logging level to WARNING and test again
    setup_logging(output_dir=tmp_path, log_level=0)
    disable_flag = get_disable_bar_flag()
    assert disable_flag is True  # CRITICAL level should disable the bar
    assert tqdm_logger.getEffectiveLevel() == logging.CRITICAL