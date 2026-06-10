"""Segmentation metrics — validated against sklearn oracles + sanity/edge cases."""
import numpy as np
import pytest
import torch
from sklearn.metrics import confusion_matrix as sk_cm, jaccard_score

from fiedler.metrics import (
    ConfusionMatrix, SegMetrics, per_class_iou, mean_iou, pixel_accuracy,
    boundary_iou, bf_score, trimap_accuracy,
)

K = 6
IGNORE = 255


def _rand_labels(n, seed, ignore_frac=0.0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, K, (n,), generator=g)
    if ignore_frac > 0:
        mask = torch.rand(n, generator=g) < ignore_frac
        t[mask] = IGNORE
    return t


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_confusion_matrix_matches_sklearn(seed):
    target = _rand_labels(5000, seed)
    pred = _rand_labels(5000, seed + 100)
    cm = ConfusionMatrix(K, ignore_index=IGNORE)
    cm.update(pred, target)
    sk = sk_cm(target.numpy(), pred.numpy(), labels=list(range(K)))
    assert np.array_equal(cm.compute().numpy(), sk)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_per_class_iou_matches_sklearn(seed):
    target = _rand_labels(8000, seed)
    pred = _rand_labels(8000, seed + 50)
    cm = ConfusionMatrix(K, IGNORE)
    cm.update(pred, target)
    ours = per_class_iou(cm.compute()).numpy()
    sk = jaccard_score(target.numpy(), pred.numpy(), labels=list(range(K)),
                       average=None, zero_division=0)
    # compare classes that are actually present (denominator > 0)
    present = ~np.isnan(ours)
    np.testing.assert_allclose(ours[present], sk[present], atol=1e-9)


def test_confusion_streams_across_batches():
    """Accumulating over many updates == one update on the concatenation."""
    cm_stream = ConfusionMatrix(K, IGNORE)
    preds, tgts = [], []
    for s in range(5):
        p, t = _rand_labels(1000, s), _rand_labels(1000, s + 200)
        cm_stream.update(p, t)
        preds.append(p); tgts.append(t)
    cm_once = ConfusionMatrix(K, IGNORE)
    cm_once.update(torch.cat(preds), torch.cat(tgts))
    assert torch.equal(cm_stream.compute(), cm_once.compute())


def test_perfect_prediction_scores_one():
    target = _rand_labels(4000, 0).reshape(40, 100)
    cm = ConfusionMatrix(K, IGNORE); cm.update(target, target)
    assert abs(mean_iou(cm.compute()) - 1.0) < 1e-9
    assert abs(pixel_accuracy(cm.compute()) - 1.0) < 1e-9
    assert abs(boundary_iou(target, target, K) - 1.0) < 1e-9
    assert abs(bf_score(target, target) - 1.0) < 1e-9
    assert abs(trimap_accuracy(target, target) - 1.0) < 1e-9


def test_ignore_index_excluded():
    """Void pixels must not affect the score: corrupting only ignore pixels is a no-op."""
    target = _rand_labels(3000, 5, ignore_frac=0.3).reshape(30, 100)
    pred = target.clone()
    pred[target == IGNORE] = 0           # wrong everywhere it's ignored
    cm = ConfusionMatrix(K, IGNORE); cm.update(pred, target)
    assert abs(mean_iou(cm.compute()) - 1.0) < 1e-9


def test_absent_class_is_nan_not_zero():
    # only classes 0 and 1 appear -> classes 2..5 must be NaN (excluded), not 0
    target = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, 0, 1, 1])
    cm = ConfusionMatrix(K, IGNORE); cm.update(pred, target)
    iou = per_class_iou(cm.compute())
    assert not torch.isnan(iou[0]) and not torch.isnan(iou[1])
    assert torch.isnan(iou[2:]).all()


def test_boundary_metrics_drop_when_misaligned():
    """A shifted prediction keeps high region mIoU but loses boundary score."""
    target = torch.zeros(64, 64, dtype=torch.long)
    target[:, 32:] = 1                      # vertical boundary at col 32
    pred = torch.zeros(64, 64, dtype=torch.long)
    pred[:, 36:] = 1                        # boundary shifted 4px
    assert bf_score(pred, target, tol=1) < 0.95
    assert boundary_iou(pred, target, K) < boundary_iou(target, target, K)


def test_segmetrics_wrapper_keys():
    sm = SegMetrics(K, IGNORE)
    sm.update(_rand_labels(2000, 1), _rand_labels(2000, 1))
    out = sm.compute()
    for key in ("mIoU", "pixel_acc", "mean_acc", "fwIoU", "per_class_iou"):
        assert key in out
    assert len(out["per_class_iou"]) == K
