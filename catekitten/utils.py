from sklearn.metrics import accuracy_score


WEIGHT_M = 1.2
WEIGHT_S = 1.3
WEIGHT_D = 1.4


def arena_accuracy_score(y_true, y_pred, category='b'):
    """Evaluation method for Kakao Arena"""
    score = accuracy_score(y_true[y_true != 0], y_pred[y_true != 0])
    if category == 'm':
        score *= WEIGHT_M
    if category == 's':
        score *= WEIGHT_S
    if category == 'd':
        score *= WEIGHT_D

    return score


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