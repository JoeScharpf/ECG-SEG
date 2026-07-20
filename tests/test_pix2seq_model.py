"""Unit tests for the Pix2Seq model: grammar-constrained decode, coordinate loss,
positional encoding, and the input-validation guards (no GPU required).

These exercise structural guarantees that hold for a randomly-initialized model,
because grammar-constrained decoding enforces them by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "semi-seg-ecg" / "src"
sys.path.insert(0, str(SRC))

import models.backbones as backbones  # noqa: E402
from models.pix2seq.model import Pix2SeqModel  # noqa: E402
from models.pix2seq.tokenizer import SegmentTokenizer  # noqa: E402

SIGNAL_LEN = 256
NUM_BINS = 16
MAX_SEGMENTS = 6


def _build_model(**overrides) -> Pix2SeqModel:
    torch.manual_seed(0)
    tok = SegmentTokenizer(
        signal_length=SIGNAL_LEN, num_bins=NUM_BINS, max_segments=MAX_SEGMENTS
    )
    backbone = backbones.resnet18(num_leads=1)
    kwargs = dict(
        d_model=32,
        nhead=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        encoder_channels=512,
        num_classes=4,
    )
    kwargs.update(overrides)
    return Pix2SeqModel(backbone=backbone, tokenizer=tok, **kwargs)


def _memory(model: Pix2SeqModel, batch: int = 4) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model.encode(torch.randn(batch, 1, SIGNAL_LEN))


def _strip(tokens, model):
    """Drop BOS/PAD, stop at EOS -> raw content tokens."""
    out = []
    for t in tokens:
        t = int(t)
        if t in (model.bos_id, model.pad_id):
            continue
        if t == model.eos_id:
            break
        out.append(t)
    return out


def _assert_raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError, none raised")


def test_decode_min_segments_validation():
    _assert_raises(lambda: _build_model(decode_min_segments=MAX_SEGMENTS + 1))
    _assert_raises(lambda: _build_model(decode_min_segments=-1))
    # In-range value builds fine.
    _build_model(decode_min_segments=MAX_SEGMENTS)


def test_class_range_invariant():
    model = _build_model()
    # bg has no token; foreground classes P/QRS/T -> tokens 3,4,5; coords start at 6.
    assert (model.class_lo, model.class_hi, model.coord_lo) == (3, 6, 6)
    assert model.class_hi - model.class_lo == 3


def test_sinusoidal_pe_odd_dim():
    for dim in (31, 32, 33):
        pe = Pix2SeqModel._sinusoidal_pe(10, dim, torch.device("cpu"), torch.float32)
        assert pe.shape == (10, dim)
        assert torch.isfinite(pe).all()


def test_max_len_clamped_no_pos_embed_overflow():
    model = _build_model()
    memory = _memory(model)
    # Oversized max_len must be clamped so pos_embed is never indexed out of range.
    tokens = model.generate(memory, max_len=10_000)
    assert tokens.shape == (memory.size(0), model.max_seq_len)


def test_constrained_output_is_wellformed_triples():
    model = _build_model()
    memory = _memory(model)
    tokens = model.generate(memory, constrained=True, chronological=False)
    for row in tokens.tolist():
        content = _strip(row, model)
        # No dangling/partial triple: content is exactly a whole number of triples.
        assert len(content) % 3 == 0
        for i in range(0, len(content), 3):
            c, s, e = content[i], content[i + 1], content[i + 2]
            assert model.class_lo <= c < model.class_hi
            assert model.coord_lo <= s < model.coord_hi
            assert model.coord_lo <= e < model.coord_hi
            assert (e - model.coord_lo) >= (s - model.coord_lo)  # offset >= onset


def test_constrained_chronological_ordering():
    model = _build_model()
    memory = _memory(model)
    tokens = model.generate(memory, constrained=True, chronological=True)
    for row in tokens.tolist():
        content = _strip(row, model)
        prev_offset = 0
        for i in range(0, len(content), 3):
            onset = content[i + 1] - model.coord_lo
            offset = content[i + 2] - model.coord_lo
            assert onset >= prev_offset  # onset >= previous segment's offset
            assert offset >= onset
            prev_offset = offset


def test_unconstrained_decode_runs():
    model = _build_model()
    memory = _memory(model)
    tokens = model.generate(memory, constrained=False)
    assert tokens.shape == (memory.size(0), model.max_seq_len)


def test_finished_rows_are_padded():
    model = _build_model()
    memory = _memory(model)
    tokens = model.generate(memory, constrained=True)
    for row in tokens.tolist():
        if model.eos_id in row:
            eos_idx = row.index(model.eos_id)
            # Everything after the first EOS must be PAD.
            assert all(t == model.pad_id for t in row[eos_idx + 1:])


def test_coord_aux_loss_finite_and_differentiable():
    model = _build_model(coord_loss="soft", coord_loss_weight=1.0)
    torch.manual_seed(1)
    logits = torch.randn(2, 6, model.vocab_size, requires_grad=True)
    # Target row with two coordinate tokens so the coord mask is non-empty.
    tgt = torch.full((2, 6), model.pad_id, dtype=torch.long)
    tgt[0, 0] = model.coord_lo + 3
    tgt[0, 1] = model.coord_lo + 7
    tgt[1, 0] = model.coord_lo + 1
    aux = model._coord_aux_loss(logits, tgt)
    assert torch.isfinite(aux) and aux.item() >= 0.0
    aux.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_expected_l1_coord_loss_differentiable():
    model = _build_model(coord_loss="expected_l1", coord_loss_weight=1.0)
    torch.manual_seed(2)
    logits = torch.randn(1, 4, model.vocab_size, requires_grad=True)
    tgt = torch.full((1, 4), model.pad_id, dtype=torch.long)
    tgt[0, 0] = model.coord_lo + 5
    aux = model._coord_aux_loss(logits, tgt)
    assert torch.isfinite(aux)
    aux.backward()
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0.0


def test_memory_pos_encoding_shapes():
    for pe in ("none", "sinusoidal", "learned"):
        model = _build_model(memory_pos_encoding=pe, max_memory_len=512)
        memory = _memory(model)
        assert memory.dim() == 3 and memory.size(-1) == model.d_model


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("model tests passed")
