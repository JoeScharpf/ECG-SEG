"""Smoke tests for the UNetHead decode head (no GPU required).

Verifies the head fuses all encoder scales, produces logits at the full signal
length through the EncoderDecoder wrapper, and is differentiable end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[1] / "semi-seg-ecg" / "src"
sys.path.insert(0, str(SRC))

import models.backbones as backbones  # noqa: E402
from models.decode_heads import UNetHead  # noqa: E402
from models.encoder_decoder import EncoderDecoder  # noqa: E402

SIGNAL_LEN = 2500
NUM_CLASSES = 4


def _build_model() -> EncoderDecoder:
    torch.manual_seed(0)
    backbone = backbones.resnet18(
        num_leads=1,
        num_stages=4,
        out_indices=[0, 1, 2, 3],
        strides=[1, 2, 2, 2],
    )
    head = UNetHead(
        in_channels=[64, 128, 256, 512],
        channels=[256, 128, 64],
        num_classes=NUM_CLASSES,
        align_corners=False,
    )
    return EncoderDecoder(
        backbone=backbone,
        decode_head=head,
        decode_head_loss=nn.CrossEntropyLoss(),
    )


def test_backbone_emits_four_scales():
    backbone = backbones.resnet18(
        num_leads=1, num_stages=4, out_indices=[0, 1, 2, 3], strides=[1, 2, 2, 2]
    )
    feats = backbone(torch.randn(2, 1, SIGNAL_LEN))
    assert len(feats) == 4
    channels = [f.shape[1] for f in feats]
    assert channels == [64, 128, 256, 512]
    # Lengths are monotonically decreasing (coarse-to-fine ordering).
    lengths = [f.shape[-1] for f in feats]
    assert lengths == sorted(lengths, reverse=True)


def test_seg_logits_shape():
    model = _build_model()
    model.eval()
    out = model(torch.randn(2, 1, SIGNAL_LEN), return_loss=False)
    assert out["seg_logits"].shape == (2, NUM_CLASSES, SIGNAL_LEN)


def test_forward_backward_runs():
    model = _build_model()
    model.train()
    inputs = torch.randn(3, 1, SIGNAL_LEN)
    labels = torch.randint(0, NUM_CLASSES, (3, SIGNAL_LEN))
    out = model(inputs, labels, return_loss=True)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_head_validates_channel_length():
    import pytest

    with pytest.raises(AssertionError):
        UNetHead(
            in_channels=[64, 128, 256, 512],
            channels=[128, 64],  # too short: must be len(in_channels) - 1
            num_classes=NUM_CLASSES,
        )
