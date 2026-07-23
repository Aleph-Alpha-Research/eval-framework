import logging

logger = logging.getLogger(__name__)


def get_disable_bar_flag() -> bool:
    return logger.getEffectiveLevel() >= logging.WARNING
