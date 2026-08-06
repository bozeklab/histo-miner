"""Unit tests for the segmentation metrics in histo_miner.evaluations.

Values here are hand-computed. get_fast_pq requires contiguous instance ids
(1..N), which every fixture below respects.
"""
import numpy as np
import pytest

from histo_miner.evaluations import get_fast_pq, remap_label

# --------------------------------------------------------------------------- #
# remap_label
# --------------------------------------------------------------------------- #

def test_remap_label_makes_ids_contiguous():
    pred = np.array([[0, 2, 2],
                     [0, 4, 4],
                     [6, 6, 0]])
    out = remap_label(pred)
    assert sorted(np.unique(out)) == [0, 1, 2, 3]


def test_remap_label_preserves_first_seen_order():
    pred = np.array([[2, 0],
                     [0, 5]])
    out = remap_label(pred)
    assert out[0, 0] == 1   # id 2 came first in np.unique order
    assert out[1, 1] == 2


def test_remap_label_by_size_gives_the_largest_instance_id_one():
    pred = np.array([[2, 0, 0],
                     [0, 5, 5],
                     [0, 5, 5]])          # id 2 -> 1 px, id 5 -> 4 px
    out = remap_label(pred, by_size=True)
    assert out[1, 1] == 1                 # largest becomes 1
    assert out[0, 0] == 2


def test_remap_label_returns_background_only_input_unchanged():
    pred = np.zeros((3, 3), dtype=int)
    assert np.array_equal(remap_label(pred), pred)


# --------------------------------------------------------------------------- #
# get_fast_pq
# --------------------------------------------------------------------------- #

def _blank():
    return np.zeros((5, 5), dtype=int)


def test_pq_is_one_when_prediction_equals_ground_truth():
    true = _blank()
    true[0:2, 0:2] = 1
    true[3:5, 3:5] = 2
    pred = true.copy()

    (dq, sq, pq), (paired_true, paired_pred, unpaired_true, unpaired_pred) = \
        get_fast_pq(true, pred)

    assert dq == pytest.approx(1.0)
    # sq divides by (tp + 1e-6), so a perfect match is marginally below 1.0
    assert sq == pytest.approx(1.0, abs=1e-5)
    assert pq == pytest.approx(1.0, abs=1e-5)
    assert len(paired_true) == 2
    assert list(unpaired_true) == []
    assert list(unpaired_pred) == []


def test_pq_is_zero_when_instances_do_not_overlap():
    true = _blank()
    true[0:2, 0:2] = 1
    pred = _blank()
    pred[3:5, 3:5] = 1          # disjoint from the ground truth instance

    (dq, sq, pq), (_, _, unpaired_true, unpaired_pred) = get_fast_pq(true, pred)

    assert dq == 0.0
    assert sq == 0.0
    assert pq == 0.0
    assert list(unpaired_true) == [1]      # one false negative
    assert list(unpaired_pred) == [1]      # one false positive


def test_sq_equals_the_iou_of_a_single_matched_pair():
    """true = 4 px, pred = 5 px, intersection = 4  ->  IoU = 4/5 = 0.8"""
    true = _blank()
    true[0:2, 0:2] = 1
    pred = _blank()
    pred[0:2, 0:2] = 1
    pred[0, 2] = 1

    (dq, sq, pq), (paired_true, _, _, _) = get_fast_pq(true, pred)

    assert len(paired_true) == 1
    assert dq == pytest.approx(1.0)        # tp=1, fp=0, fn=0
    assert sq == pytest.approx(0.8, abs=1e-5)
    assert pq == pytest.approx(0.8, abs=1e-5)


def test_dq_penalises_a_missed_instance():
    """tp=1, fp=0, fn=1  ->  dq = 1 / (1 + 0.5) = 0.6667"""
    true = _blank()
    true[0:2, 0:2] = 1
    true[3:5, 3:5] = 2
    pred = _blank()
    pred[0:2, 0:2] = 1                     # second instance not predicted

    (dq, _, _), (_, _, unpaired_true, unpaired_pred) = get_fast_pq(true, pred)

    assert dq == pytest.approx(1.0 / 1.5)
    assert list(unpaired_true) == [2]
    assert list(unpaired_pred) == []


def test_overlap_below_threshold_is_not_paired():
    """intersection = 1 px, union = 7 px  ->  IoU = 0.143 < match_iou = 0.5"""
    true = _blank()
    true[0:2, 0:2] = 1
    pred = _blank()
    pred[1:3, 1:3] = 1

    (dq, sq, pq), (paired_true, _, _, _) = get_fast_pq(true, pred)

    assert len(paired_true) == 0
    assert dq == 0.0
    assert pq == 0.0


def test_low_match_iou_uses_munkres_and_accepts_the_weak_pair():
    """Same geometry as above, but match_iou below the IoU value pairs them."""
    true = _blank()
    true[0:2, 0:2] = 1
    pred = _blank()
    pred[1:3, 1:3] = 1

    (dq, sq, pq), (paired_true, _, _, _) = get_fast_pq(true, pred, match_iou=0.1)

    assert len(paired_true) == 1
    assert sq == pytest.approx(1.0 / 7.0, abs=1e-5)


def test_negative_match_iou_is_rejected():
    true = _blank()
    true[0:2, 0:2] = 1
    with pytest.raises(AssertionError):
        get_fast_pq(true, true, match_iou=-0.1)
