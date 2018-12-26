from sklearn.metrics import accuracy_score
from collections import defaultdict
import fire
import h5py
import numpy as np
import six
from six.moves import zip, cPickle


WEIGHT_B = 1.0
WEIGHT_M = 1.2
WEIGHT_S = 1.3
WEIGHT_D = 1.4


def arena_accuracy_score(
    b_pred=None, b_gt=None, m_pred=None, m_gt=None,
    s_pred=None, s_gt=None, d_pred=None, d_gt=None, weighted=True):

    n_classes = 0
    preds = [b_pred, m_pred, s_pred, d_pred]
    gts = [b_gt, m_gt, s_gt, d_gt]
    constants = [WEIGHT_B, WEIGHT_M, WEIGHT_S, WEIGHT_D]
    scores = []

    for pred, gt, constant in zip(preds, gts, constants):
        if (pred is not None) & (gt is not None):
            # Remove -1 index of ground truth and prediction from evaluation
            pred = np.delete(pred, np.where(gt == -1))
            gt = np.delete(gt, np.where(gt == -1))

            n_classes += 1
            score = accuracy_score(pred, gt)
            if weighted:
                score *= constant
            scores.append(score)
    
    return np.array(scores) / n_classes


def evaluate(predict_path, data_path, div, y_vocab_path):
    h = h5py.File(data_path, 'r')[div]
    y_vocab = cPickle.loads(open(y_vocab_path).read())
    inv_y_vocab = {v: k for k, v in six.iteritems(y_vocab)}
    fin = open(predict_path, 'rb')
    hit, n = defaultdict(lambda: 0), defaultdict(lambda: 0)
    print('loading ground-truth...')
    CATE = np.argmax(h['cate'], axis=1)
    for p, y in zip(fin, CATE):
        pid, b, m, s, d = p.split('\t')
        b, m, s, d = list(map(int, [b, m, s, d]))
        gt = list(map(int, inv_y_vocab[y].split('>')))
        for depth, _p, _g in zip(['b', 'm', 's', 'd'],
                                 [b, m, s, d],
                                 gt):
            if _g == -1:
                continue
            n[depth] = n.get(depth, 0) + 1
            if _p == _g:
                hit[depth] = hit.get(depth, 0) + 1
    for d in ['b', 'm', 's', 'd']:
        if n[d] > 0:
            print('%s-Accuracy: %.3f(%s/%s)' % (d, hit[d] / float(n[d]), hit[d], n[d]))
    score = sum([hit[d] / float(n[d]) * w
                 for d, w in zip(['b', 'm', 's', 'd'],
                                 [1.0, 1.2, 1.3, 1.4])]) / 4.0
    print('score: %.3f' % score)


if __name__ == '__main__':
    fire.Fire({'evaluate': evaluate})
