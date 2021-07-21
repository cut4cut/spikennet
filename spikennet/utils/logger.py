import logging


def get_logger():
    formatter = logging.Formatter('%(asctime)s %(name)-22s %(levelname)-8s %(message)s')
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger
