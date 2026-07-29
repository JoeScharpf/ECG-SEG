"""Unit tests for soft boundary weight helper."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "semi-seg-ecg" / "src"
sys.path.insert(0, str(SRC))

from utils.boundary_weights import soft_boundary_weight  # noqa: E402


def test_soft_boundary_weight_peaks_at_class_change():
    logits = torch.zeros(1, 4, 200)
    logits[0, 0, :100] = 5.0
    logits[0, 1, 100:] = 5.0
    prob = logits.softmax(dim=1)
    w = soft_boundary_weight(prob, band=4)
    assert w.shape == (1, 200)
    assert float(w[0, 100]) > 0.5
    assert float(w[0, 10]) < 0.2
    assert 0.0 <= float(w.min()) <= float(w.max()) <= 1.0


def test_soft_boundary_weight_band_zero():
    logits = torch.zeros(2, 4, 64)
    logits[:, 2, 32:] = 4.0
    logits[:, 0, :32] = 4.0
    w = soft_boundary_weight(logits.softmax(dim=1), band=0)
    assert w.shape == (2, 64)
    assert float(w[:, 32].mean()) > float(w[:, 5].mean())
