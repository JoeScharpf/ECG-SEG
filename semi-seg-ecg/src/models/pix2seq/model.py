# Copyright (c) ECG-SEG. Minimal Pix2Seq-style model for ECG delineation.

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import models.backbones as backbones
from models.pix2seq.tokenizer import SegmentTokenizer


class Pix2SeqModel(nn.Module):
    """ResNet-1D encoder + Transformer decoder that emits segment tokens."""

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: SegmentTokenizer,
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        encoder_channels: int = 512,
        num_classes: int = 4,
        memory_pos_encoding: str = "none",
        max_memory_len: int = 1024,
        decode_constrained: bool = False,
        decode_min_segments: int = 0,
        decode_chronological: bool = True,
        coord_loss: str = "none",
        coord_loss_weight: float = 0.0,
        coord_soft_sigma: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.num_classes = num_classes
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = tokenizer.max_seq_len
        if tokenizer.num_classes != num_classes:
            raise ValueError(
                f"tokenizer.num_classes={tokenizer.num_classes} != "
                f"model num_classes={num_classes}"
            )

        # Token-id ranges (from Vocab) used by grammar-constrained decoding and
        # the coordinate-aware loss.
        vocab = tokenizer.vocab
        self.pad_id = vocab.pad
        self.bos_id = vocab.bos
        self.eos_id = vocab.eos
        self.class_lo = vocab.class_offset
        self.class_hi = vocab.class_offset + (num_classes - 1)  # exclusive
        self.coord_lo = vocab.coord_offset
        self.coord_hi = vocab.coord_offset + tokenizer.num_bins  # exclusive
        self.num_bins = tokenizer.num_bins

        # Invariant: the derived class-token range must match the tokenizer's
        # actual foreground (non-background) classes, so num_classes-1 is not a
        # silent assumption. Background must have no segment token.
        n_seg_classes = len(tokenizer.wave_classes)
        if self.class_hi - self.class_lo != n_seg_classes:
            raise ValueError(
                f"class-token range [{self.class_lo}, {self.class_hi}) implies "
                f"{self.class_hi - self.class_lo} foreground classes, but tokenizer "
                f"has {n_seg_classes} (wave_classes={tokenizer.wave_classes})."
            )
        if self.class_hi != self.coord_lo:
            raise ValueError(
                f"class range must end where coords begin: class_hi={self.class_hi}, "
                f"coord_lo={self.coord_lo}."
            )

        if memory_pos_encoding not in ("none", "sinusoidal", "learned"):
            raise ValueError(
                f"memory_pos_encoding must be none|sinusoidal|learned, got {memory_pos_encoding}"
            )
        if coord_loss not in ("none", "soft", "expected_l1"):
            raise ValueError(
                f"coord_loss must be none|soft|expected_l1, got {coord_loss}"
            )
        if not 0 <= decode_min_segments <= tokenizer.max_segments:
            raise ValueError(
                f"decode_min_segments must be in [0, max_segments="
                f"{tokenizer.max_segments}], got {decode_min_segments}; otherwise "
                "EOS can never become legal before the length limit."
            )
        self.memory_pos_encoding = memory_pos_encoding
        self.max_memory_len = max_memory_len
        self.decode_constrained = decode_constrained
        self.decode_min_segments = decode_min_segments
        self.decode_chronological = decode_chronological
        self.coord_loss = coord_loss
        self.coord_loss_weight = coord_loss_weight
        self.coord_soft_sigma = coord_soft_sigma

        self.input_proj = nn.Conv1d(encoder_channels, d_model, kernel_size=1)
        self.token_embed = nn.Embedding(
            self.vocab_size, d_model, padding_idx=tokenizer.vocab.pad
        )
        self.pos_embed = nn.Embedding(self.max_seq_len, d_model)
        if memory_pos_encoding == "learned":
            self.memory_pos_embed = nn.Embedding(max_memory_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.lm_head = nn.Linear(d_model, self.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.pad)

    def no_weight_decay(self):
        params = {"token_embed.weight", "pos_embed.weight"}
        if self.memory_pos_encoding == "learned":
            params.add("memory_pos_embed.weight")
        return params

    @staticmethod
    def _sinusoidal_pe(length: int, dim: int, device, dtype) -> Tensor:
        """Standard sinusoidal positional encoding, shape (length, dim)."""
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Slice the cosine term so odd `dim` (where the sin/cos column counts
        # differ by one) does not raise a shape-mismatch.
        pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].size(1)]
        return pe.to(dtype)

    def _add_memory_pos(self, memory: Tensor) -> Tensor:
        """Add positional encoding to encoder memory (B, L, d_model)."""
        if self.memory_pos_encoding == "none":
            return memory
        length = memory.size(1)
        if self.memory_pos_encoding == "sinusoidal":
            pe = self._sinusoidal_pe(length, self.d_model, memory.device, memory.dtype)
            return memory + pe.unsqueeze(0)
        # learned
        if length > self.max_memory_len:
            raise ValueError(
                f"memory length {length} exceeds max_memory_len {self.max_memory_len}; "
                "raise max_memory_len in the pix2seq config."
            )
        idx = torch.arange(length, device=memory.device)
        return memory + self.memory_pos_embed(idx).unsqueeze(0)

    def encode(self, inputs: Tensor) -> Tensor:
        """Encode ECG (B, 1, T) to memory (B, L, d_model).

        Assumes fixed-length batches (SemiSegECG resamples/crops to signal_length),
        so no memory_key_padding_mask is applied in the decoder. Positional encoding
        is added to the memory when enabled (see memory_pos_encoding).
        """
        feats = self.backbone(inputs)[-1]  # (B, C, L)
        memory = self.input_proj(feats).transpose(1, 2)  # (B, L, d_model)
        memory = self._add_memory_pos(memory)
        return memory

    def _embed_tokens(self, tokens: Tensor) -> Tensor:
        # tokens: (B, S)
        b, s = tokens.shape
        positions = torch.arange(s, device=tokens.device).unsqueeze(0).expand(b, -1)
        return self.token_embed(tokens) + self.pos_embed(positions)

    def forward_tokens(
        self,
        memory: Tensor,
        tgt_tokens: Tensor,
    ) -> Tensor:
        """Teacher-forced decode. tgt_tokens: (B, S) including BOS...; returns logits (B, S, V)."""
        s = tgt_tokens.size(1)
        tgt = self._embed_tokens(tgt_tokens)
        # Causal mask (PyTorch 1.11-compatible): True = blocked
        causal = torch.triu(
            torch.ones(s, s, device=tgt.device, dtype=torch.bool),
            diagonal=1,
        )
        pad_mask = tgt_tokens.eq(self.tokenizer.vocab.pad)
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
            # No memory_key_padding_mask: encoder features are fixed-length for LUDB.
        )
        return self.lm_head(out)

    def _grammar_masked_logits(
        self,
        next_logits: Tensor,
        triple_pos: Tensor,
        onset_bin: Tensor,
        prev_offset_bin: Tensor,
        seg_count: Tensor,
        min_segments: int,
        remaining_steps: int,
        chronological: bool,
    ) -> Tensor:
        """Restrict next-token logits to the legal set given the grammar state.

        Grammar: BOS -> (CLASS ONSET OFFSET)* -> EOS. States:
          triple_pos == 0: expect a class token, or EOS. A new triple (class) is
            only allowed when at least 3 slots remain, so a started triple can
            always be completed; otherwise EOS is forced (never a dangling class).
          triple_pos == 1: expect an onset coordinate (>= previous offset when
            ``chronological``).
          triple_pos == 2: expect an offset coordinate >= onset.
        Because a triple is only started with >= 3 slots free, states 1 and 2
        always have room to finish, so they never need slot reservation.
        """
        neg = torch.finfo(next_logits.dtype).min
        mask = torch.full_like(next_logits, neg)

        s0 = triple_pos == 0
        if s0.any():
            if remaining_steps >= 3:
                mask[s0, self.class_lo:self.class_hi] = 0.0
                allow_eos = s0 & (seg_count >= min_segments)
                if allow_eos.any():
                    mask[allow_eos, self.eos_id] = 0.0
            else:
                # No room for another full triple: force EOS (even if the
                # min_segments floor was not reached) to avoid a dangling class.
                mask[s0, self.eos_id] = 0.0

        s1 = triple_pos == 1
        if s1.any():
            if chronological:
                for r in torch.nonzero(s1, as_tuple=False).flatten().tolist():
                    lo = self.coord_lo + int(prev_offset_bin[r].item())
                    mask[r, lo:self.coord_hi] = 0.0
            else:
                mask[s1, self.coord_lo:self.coord_hi] = 0.0

        s2 = triple_pos == 2
        if s2.any():
            for r in torch.nonzero(s2, as_tuple=False).flatten().tolist():
                # Allow offset >= onset (equal permitted to avoid an empty set at
                # the last bin; equal bins are a well-defined single-sample
                # segment under the tokenizer's inclusive-boundary convention).
                lo = self.coord_lo + int(onset_bin[r].item())
                mask[r, lo:self.coord_hi] = 0.0

        return next_logits + mask

    @torch.no_grad()
    def generate(
        self,
        memory: Tensor,
        max_len: Optional[int] = None,
        constrained: Optional[bool] = None,
        min_segments: Optional[int] = None,
        chronological: Optional[bool] = None,
    ) -> Tensor:
        """Autoregressive decode to token ids (B, max_seq_len).

        ``max_len`` limits how many decode *steps* run and is clamped to
        ``self.max_seq_len`` so decoder positional embeddings can never be indexed
        out of range. The returned tensor is always padded/truncated to
        ``self.max_seq_len`` for consistent batch_decode. No KV cache: each step
        recomputes attention over the full prefix (fine for short ECG segment
        sequences; O(n^2) if max_seq_len grows large).

        When ``constrained`` is True, logits are grammar-masked at each step so the
        output is a sequence of complete (class, onset, offset) triples optionally
        followed by EOS. A new triple is only started when >= 3 slots remain, so the
        stream never ends with a partial/dangling triple. When ``chronological`` is
        True, each onset is also constrained to be >= the previous segment's offset.
        """
        constrained = self.decode_constrained if constrained is None else constrained
        min_segments = self.decode_min_segments if min_segments is None else min_segments
        chronological = self.decode_chronological if chronological is None else chronological
        max_len = min(max_len or self.max_seq_len, self.max_seq_len)
        b = memory.size(0)
        device = memory.device
        bos = self.bos_id
        eos = self.eos_id
        pad = self.pad_id

        tokens = torch.full((b, 1), bos, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        triple_pos = torch.zeros(b, dtype=torch.long, device=device)
        onset_bin = torch.zeros(b, dtype=torch.long, device=device)
        prev_offset_bin = torch.zeros(b, dtype=torch.long, device=device)
        seg_count = torch.zeros(b, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward_tokens(memory, tokens)
            next_logits = logits[:, -1, :].float()
            if constrained:
                remaining_steps = max_len - tokens.size(1)
                next_logits = self._grammar_masked_logits(
                    next_logits, triple_pos, onset_bin, prev_offset_bin,
                    seg_count, min_segments, remaining_steps, chronological,
                )
            next_token = next_logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, pad), next_token)

            active = ~finished
            is_class = active & (next_token >= self.class_lo) & (next_token < self.class_hi)
            is_coord = active & (next_token >= self.coord_lo) & (next_token < self.coord_hi)
            is_eos = active & next_token.eq(eos)

            set_onset = is_coord & (triple_pos == 1)
            complete = is_coord & (triple_pos == 2)
            onset_bin = torch.where(set_onset, next_token - self.coord_lo, onset_bin)
            prev_offset_bin = torch.where(complete, next_token - self.coord_lo, prev_offset_bin)
            seg_count = seg_count + complete.long()

            new_tp = triple_pos.clone()
            new_tp = torch.where(is_class, torch.ones_like(new_tp), new_tp)
            new_tp = torch.where(set_onset, torch.full_like(new_tp, 2), new_tp)
            new_tp = torch.where(complete, torch.zeros_like(new_tp), new_tp)
            triple_pos = new_tp

            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | is_eos
            if bool(finished.all()):
                break

        # Always return length max_seq_len (see docstring).
        if tokens.size(1) < self.max_seq_len:
            pad_len = self.max_seq_len - tokens.size(1)
            tokens = F.pad(tokens, (0, pad_len), value=pad)
        else:
            tokens = tokens[:, : self.max_seq_len]
        return tokens

    def tokens_to_seg_logits(self, tokens: Tensor, seq_len: int) -> Tensor:
        """Rasterize tokens to one-hot-style logits (B, num_classes, T) for MeanIoU."""
        masks = self.tokenizer.batch_decode(tokens)  # (B, T)
        if masks.size(1) != seq_len:
            if masks.size(1) > seq_len:
                masks = masks[:, :seq_len]
            else:
                masks = F.pad(masks, (0, seq_len - masks.size(1)), value=0)

        if int(masks.min()) < 0 or int(masks.max()) >= self.num_classes:
            raise ValueError(
                f"Decoded mask class ids out of range [0, {self.num_classes - 1}]: "
                f"min={int(masks.min())}, max={int(masks.max())}"
            )
        # Hard one-hot scaled as logits so evaluate()'s softmax → argmax path works.
        logits = F.one_hot(masks, num_classes=self.num_classes)
        logits = logits.float().movedim(-1, 1) * 10.0  # (B, C, T)
        return logits

    def _coord_aux_loss(self, logits: Tensor, tgt_out: Tensor) -> Tensor:
        """Distance-aware auxiliary loss on coordinate positions only.

        ``soft``: Gaussian soft targets over neighboring bins (CE/KL).
        ``expected_l1``: L1 between the soft-argmax expected bin and the true bin
        (differentiable, unlike L1 on a hard argmax).
        """
        coord_mask = (tgt_out >= self.coord_lo) & (tgt_out < self.coord_hi)
        if not bool(coord_mask.any()):
            return logits.new_zeros(())

        sel = logits[coord_mask]  # (N, V)
        coord_logits = sel[:, self.coord_lo:self.coord_hi].float()  # (N, num_bins)
        tgt_bins = (tgt_out[coord_mask] - self.coord_lo).long()  # (N,)
        bins = torch.arange(self.num_bins, device=logits.device, dtype=torch.float32)

        if self.coord_loss == "soft":
            sigma = max(self.coord_soft_sigma, 1e-6)
            q = torch.exp(-((bins[None, :] - tgt_bins[:, None].float()) ** 2) / (2 * sigma ** 2))
            q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            logp = F.log_softmax(coord_logits, dim=-1)
            return -(q * logp).sum(dim=-1).mean()

        # expected_l1
        p = F.softmax(coord_logits, dim=-1)
        expected_bin = (p * bins[None, :]).sum(dim=-1)
        return (expected_bin - tgt_bins.float()).abs().mean()

    def forward(
        self,
        inputs: Tensor,
        labels: Optional[Tensor] = None,
        return_loss: bool = False,
        decode: bool = False,
    ) -> dict:
        """
        Args:
            inputs: (B, 1, T) ECG
            labels: (B, T) multi-class masks
            return_loss: compute token CE (requires labels)
            decode: if True, run real autoregressive generate() for seg_logits.
                    Official val/test MeanIoU must use decode=True.
                    If False, only token loss is computed (no teacher-forced
                    seg_logits) — teacher-forced argmax would be conditioned on
                    ground-truth prefixes and inflate IoU vs true AR inference.

        Eval-mode contract: any decode path (decode=True, or label-free inference)
        must be called with the module already in eval() so the *whole* path —
        backbone/BatchNorm/dropout in encode() and the decoder in generate() — runs
        deterministically. forward() no longer toggles training state internally
        (that only covered generate(), not encode(), which was misleading). The
        eval loop and diagnostics already call model.eval() before decoding.
        """
        outputs = {}
        seq_len = inputs.size(-1)
        memory = self.encode(inputs)

        if labels is not None and return_loss:
            if labels.dim() == 3:
                labels = labels.squeeze(1)
            target_tokens = self.tokenizer.batch_encode(labels, device=inputs.device)
            tgt_in = target_tokens[:, :-1]
            tgt_out = target_tokens[:, 1:]
            logits = self.forward_tokens(memory, tgt_in)
            loss = self.loss_fn(
                logits.reshape(-1, self.vocab_size),
                tgt_out.reshape(-1),
            )
            if self.coord_loss != "none" and self.coord_loss_weight > 0:
                loss = loss + self.coord_loss_weight * self._coord_aux_loss(logits, tgt_out)
            outputs["loss"] = loss
            # Intentionally do NOT build teacher-forced seg_logits here.
            # Use decode=True for any segmentation metric that should match test-time AR.

        if decode or ("seg_logits" not in outputs and not return_loss):
            # Real AR decode. generate() is already @torch.no_grad(); the caller
            # is responsible for eval() (see the eval-mode contract above).
            gen_tokens = self.generate(memory)
            outputs["seg_logits"] = self.tokens_to_seg_logits(gen_tokens, seq_len)
            outputs["tokens"] = gen_tokens

        return outputs


def build_pix2seq_from_cfg(config: dict) -> Pix2SeqModel:
    backbone_name, backbone_kwargs = list(config["backbone"].items())[0]
    assert backbone_name in backbones.__dict__, f"Unsupported backbone: {backbone_name}"
    backbone = backbones.__dict__[backbone_name](**backbone_kwargs)

    p2s = config.get("pix2seq", {})
    num_classes = p2s.get(
        "num_classes",
        config.get("metric", {}).get("num_classes", 4),
    )
    tokenizer = SegmentTokenizer(
        signal_length=config["dataset"].get("signal_length", 2500),
        num_bins=p2s.get("num_bins", 250),
        max_segments=p2s.get("max_segments", 32),
        num_classes=num_classes,
    )
    model = Pix2SeqModel(
        backbone=backbone,
        tokenizer=tokenizer,
        d_model=p2s.get("d_model", 256),
        nhead=p2s.get("nhead", 4),
        num_decoder_layers=p2s.get("num_decoder_layers", 4),
        dim_feedforward=p2s.get("dim_feedforward", 512),
        dropout=p2s.get("dropout", 0.1),
        encoder_channels=p2s.get("encoder_channels", 512),
        num_classes=num_classes,
        memory_pos_encoding=p2s.get("memory_pos_encoding", "none"),
        max_memory_len=p2s.get("max_memory_len", 1024),
        decode_constrained=p2s.get("decode_constrained", False),
        decode_min_segments=p2s.get("decode_min_segments", 0),
        decode_chronological=p2s.get("decode_chronological", True),
        coord_loss=p2s.get("coord_loss", "none"),
        coord_loss_weight=p2s.get("coord_loss_weight", 0.0),
        coord_soft_sigma=p2s.get("coord_soft_sigma", 1.0),
    )
    return model
