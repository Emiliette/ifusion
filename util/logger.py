import logging


def get_logger() -> logging.Logger:
    logger = logging.getLogger("DPS")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] >> %(message)s"))
    logger.addHandler(handler)
    return logger

