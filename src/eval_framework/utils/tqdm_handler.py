import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def safe_tqdm_write(msg: str, level: int = logging.INFO) -> None:
    if logger.isEnabledFor(level):
        tqdm.write(msg)


def get_disable_bar_flag() -> bool:
    return logger.getEffectiveLevel() >= logging.WARNING
