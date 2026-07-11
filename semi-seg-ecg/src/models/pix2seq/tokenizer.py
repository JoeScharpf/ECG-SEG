# Copyright (c) ECG-SEG. Pix2Seq-style tokenization for multi-class ECG delineation.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Vocab:
    pad: int = 0
    bos: int = 1
    eos: int = 2
    # Class tokens for P=1, QRS=2, T=3 map to class_offset + class_id
    class_offset: int = 3  # tokens 3,4,5 for classes 1,2,3
    coord_offset: int = 6  # bins start here

    def class_token(self, class_id: int) -> int:
        assert class_id in (1, 2, 3), f"Expected wave class 1-3, got {class_id}"
        return self.class_offset + (class_id - 1)

    def token_to_class(self, token: int) -> int:
        return token - self.class_offset + 1

    def is_class_token(self, token: int) -> bool:
        return self.class_offset <= token < self.coord_offset

    def coord_token(self, bin_id: int, num_bins: int) -> int:
        assert 0 <= bin_id < num_bins
        return self.coord_offset + bin_id

    def token_to_bin(self, token: int) -> int:
        return token - self.coord_offset

    def is_coord_token(self, token: int, num_bins: int) -> bool:
        return self.coord_offset <= token < self.coord_offset + num_bins

    def vocab_size(self, num_bins: int) -> int:
        return self.coord_offset + num_bins


class SegmentTokenizer:
    """Convert dense multi-class masks <-> quantized segment token sequences."""

    def __init__(
        self,
        signal_length: int = 2500,
        num_bins: int = 250,
        max_segments: int = 32,
        vocab: Vocab | None = None,
    ):
        self.signal_length = signal_length
        self.num_bins = num_bins
        self.max_segments = max_segments
        self.vocab = vocab or Vocab()
        self.max_seq_len = 1 + max_segments * 3 + 1  # BOS + triples + EOS

    @property
    def vocab_size(self) -> int:
        return self.vocab.vocab_size(self.num_bins)

    def quantize(self, index: int) -> int:
        index = int(np.clip(index, 0, self.signal_length - 1))
        bin_id = int(index * self.num_bins / self.signal_length)
        return min(bin_id, self.num_bins - 1)

    def dequantize(self, bin_id: int) -> int:
        bin_id = int(np.clip(bin_id, 0, self.num_bins - 1))
        # Map bin center back to sample index
        return int((bin_id + 0.5) * self.signal_length / self.num_bins)

    def mask_to_segments(self, mask: np.ndarray | torch.Tensor) -> List[Tuple[int, int, int]]:
        """Extract contiguous non-background segments as (class, start, end_inclusive)."""
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask).astype(np.int64).reshape(-1)
        segments: List[Tuple[int, int, int]] = []
        t = 0
        n = len(mask)
        while t < n:
            c = int(mask[t])
            if c == 0:
                t += 1
                continue
            start = t
            while t < n and int(mask[t]) == c:
                t += 1
            end = t - 1
            if c in (1, 2, 3):
                segments.append((c, start, end))
            # Ignore unexpected class ids
        return segments

    def segments_to_tokens(self, segments: Sequence[Tuple[int, int, int]]) -> List[int]:
        tokens = [self.vocab.bos]
        for class_id, start, end in segments[: self.max_segments]:
            tokens.append(self.vocab.class_token(class_id))
            tokens.append(self.vocab.coord_token(self.quantize(start), self.num_bins))
            tokens.append(self.vocab.coord_token(self.quantize(end), self.num_bins))
        tokens.append(self.vocab.eos)
        return tokens

    def tokens_to_segments(self, tokens: Sequence[int]) -> List[Tuple[int, int, int]]:
        # Drop BOS/PAD; stop at EOS
        cleaned = []
        for tok in tokens:
            tok = int(tok)
            if tok == self.vocab.pad:
                continue
            if tok == self.vocab.bos:
                continue
            if tok == self.vocab.eos:
                break
            cleaned.append(tok)

        segments: List[Tuple[int, int, int]] = []
        i = 0
        while i + 2 < len(cleaned):
            c_tok, s_tok, e_tok = cleaned[i], cleaned[i + 1], cleaned[i + 2]
            if (
                self.vocab.is_class_token(c_tok)
                and self.vocab.is_coord_token(s_tok, self.num_bins)
                and self.vocab.is_coord_token(e_tok, self.num_bins)
            ):
                class_id = self.vocab.token_to_class(c_tok)
                start = self.dequantize(self.vocab.token_to_bin(s_tok))
                end = self.dequantize(self.vocab.token_to_bin(e_tok))
                if end < start:
                    start, end = end, start
                start = int(np.clip(start, 0, self.signal_length - 1))
                end = int(np.clip(end, 0, self.signal_length - 1))
                segments.append((class_id, start, end))
                i += 3
            else:
                i += 1  # resync
        return segments

    def segments_to_mask(self, segments: Sequence[Tuple[int, int, int]]) -> np.ndarray:
        mask = np.zeros(self.signal_length, dtype=np.int64)
        for class_id, start, end in segments:
            mask[start : end + 1] = class_id
        return mask

    def encode_mask(self, mask: np.ndarray | torch.Tensor) -> List[int]:
        return self.segments_to_tokens(self.mask_to_segments(mask))

    def decode_tokens(self, tokens: Sequence[int]) -> np.ndarray:
        return self.segments_to_mask(self.tokens_to_segments(tokens))

    def batch_encode(
        self,
        masks: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode a batch of masks (B, T) into padded token ids (B, max_seq_len)."""
        if masks.dim() == 3:
            masks = masks.squeeze(1)
        masks = masks.long()
        batch = []
        for i in range(masks.size(0)):
            toks = self.encode_mask(masks[i])
            if len(toks) > self.max_seq_len:
                toks = toks[: self.max_seq_len - 1] + [self.vocab.eos]
            pad_len = self.max_seq_len - len(toks)
            toks = toks + [self.vocab.pad] * pad_len
            batch.append(toks)
        out = torch.tensor(batch, dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

    def batch_decode(self, token_batch: torch.Tensor) -> torch.Tensor:
        """Decode (B, S) token ids to dense masks (B, T)."""
        masks = []
        for i in range(token_batch.size(0)):
            masks.append(self.decode_tokens(token_batch[i].tolist()))
        return torch.tensor(np.stack(masks), dtype=torch.long, device=token_batch.device)
