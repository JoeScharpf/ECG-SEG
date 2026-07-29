"""Unit tests for baseline/eval_boundaries.py matching helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from baseline.eval_boundaries import (  # noqa: E402
    evaluate_run,
    mask_to_segments,
    match_segments,
    temporal_iou,
)


def test_mask_to_segments_basic():
    mask = np.zeros(20, dtype=np.int64)
    mask[2:5] = 1  # P
    mask[6:9] = 2  # QRS
    mask[10:15] = 3  # T
    segs = mask_to_segments(mask)
    assert segs == [(1, 2, 4), (2, 6, 8), (3, 10, 14)]


def test_match_greedy_iou():
    gt = [(2, 10, 20), (2, 40, 50)]
    pred = [(2, 12, 22), (2, 100, 110)]
    pairs, ug, up = match_segments(gt, pred, class_id=2, min_iou=0.1)
    assert len(pairs) == 1
    assert pairs[0][0] == (2, 10, 20)
    assert len(ug) == 1
    assert len(up) == 1


def test_evaluate_perfect():
    gt = np.zeros((1, 100), dtype=np.int64)
    gt[0, 10:20] = 2
    gt[0, 25:40] = 3
    pred = gt.copy()
    report = evaluate_run(pred, gt, match_iou=0.1)
    assert report["pixel_iou"]["mean"] == 1.0
    assert report["boundary_mae"]["QRS"]["onset"]["mae_samples"] == 0.0
    assert report["detection"]["n_matched_pairs"] == 2


def test_temporal_iou():
    assert temporal_iou((2, 0, 9), (2, 0, 9)) == 1.0
    assert temporal_iou((2, 0, 9), (2, 10, 19)) == 0.0


if __name__ == "__main__":
    test_mask_to_segments_basic()
    test_match_greedy_iou()
    test_evaluate_perfect()
    test_temporal_iou()
    print("ok")
