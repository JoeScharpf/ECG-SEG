"""Unit tests for Pix2Seq segment tokenizer (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "semi-seg-ecg" / "src"
sys.path.insert(0, str(SRC))

from models.pix2seq.tokenizer import SegmentTokenizer  # noqa: E402


def test_roundtrip_simple():
    tok = SegmentTokenizer(signal_length=100, num_bins=10, max_segments=8)
    mask = np.zeros(100, dtype=np.int64)
    mask[10:20] = 1  # P
    mask[30:45] = 2  # QRS
    mask[50:70] = 3  # T

    tokens = tok.encode_mask(mask)
    assert tokens[0] == tok.vocab.bos
    assert tokens[-1] == tok.vocab.eos

    recon = tok.decode_tokens(tokens)
    # Quantization is lossy; check class presence and rough overlap
    assert set(np.unique(recon)) <= {0, 1, 2, 3}
    assert (recon == 1).sum() > 0
    assert (recon == 2).sum() > 0
    assert (recon == 3).sum() > 0


def test_batch_encode_decode():
    tok = SegmentTokenizer(signal_length=2500, num_bins=250, max_segments=32)
    masks = torch.zeros(2, 2500, dtype=torch.long)
    masks[0, 100:200] = 1
    masks[0, 400:500] = 2
    masks[1, 800:900] = 3

    tokens = tok.batch_encode(masks)
    assert tokens.shape == (2, tok.max_seq_len)
    decoded = tok.batch_decode(tokens)
    assert decoded.shape == (2, 2500)
    assert (decoded[0] == 1).any()
    assert (decoded[1] == 3).any()


if __name__ == "__main__":
    test_roundtrip_simple()
    test_batch_encode_decode()
    print("tokenizer tests passed")
