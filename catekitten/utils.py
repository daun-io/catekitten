import logging


def lowercase(data):
    return data.lower()


def gen_batch(iterable, n=1):
    """Yields batches.
    Args:
        iterable: iterable object.
        n (int, optional): Defaults to 1. batchsize.
    """
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def get_logger():
    logger = logging.getLogger('catekitten')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # stream handler
    sh_debug = logging.StreamHandler()
    sh_debug.setLevel(logging.DEBUG)
    sh_debug.setFormatter(formatter)
    # add both
    logger.addHandler(sh_debug)
    return logger


logger = get_logger()
info = logger.info
