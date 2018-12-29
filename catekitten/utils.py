from sklearn.metrics import accuracy_score


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