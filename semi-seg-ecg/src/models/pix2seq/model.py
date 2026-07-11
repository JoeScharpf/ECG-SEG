# Copyright (c) ECG-SEG. Minimal Pix2Seq-style model for ECG delineation.

from __future__ import annotations

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

        self.input_proj = nn.Conv1d(encoder_channels, d_model, kernel_size=1)
        self.token_embed = nn.Embedding(
            self.vocab_size, d_model, padding_idx=tokenizer.vocab.pad
        )
        self.pos_embed = nn.Embedding(self.max_seq_len, d_model)

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
        return {"token_embed.weight", "pos_embed.weight"}

    def encode(self, inputs: Tensor) -> Tensor:
        """Encode ECG (B, 1, T) to memory (B, L, d_model).

        Assumes fixed-length batches (SemiSegECG resamples/crops to signal_length),
        so no memory_key_padding_mask is applied in the decoder.
        """
        feats = self.backbone(inputs)[-1]  # (B, C, L)
        memory = self.input_proj(feats).transpose(1, 2)  # (B, L, d_model)
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

    @torch.no_grad()
    def generate(self, memory: Tensor, max_len: Optional[int] = None) -> Tensor:
        """Autoregressive decode to token ids (B, max_seq_len).

        ``max_len`` only limits how many decode *steps* run. The returned tensor is
        always padded/truncated to ``self.max_seq_len`` for consistent batch_decode.
        No KV cache: each step recomputes attention over the full prefix (fine for
        short ECG segment sequences; O(n^2) if max_seq_len grows large).
        """
        max_len = max_len or self.max_seq_len
        b = memory.size(0)
        device = memory.device
        bos = self.tokenizer.vocab.bos
        eos = self.tokenizer.vocab.eos
        pad = self.tokenizer.vocab.pad

        tokens = torch.full((b, 1), bos, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward_tokens(memory, tokens)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, pad), next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos)
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
            outputs["loss"] = self.loss_fn(
                logits.reshape(-1, self.vocab_size),
                tgt_out.reshape(-1),
            )
            # Intentionally do NOT build teacher-forced seg_logits here.
            # Use decode=True for any segmentation metric that should match test-time AR.

        if decode:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                gen_tokens = self.generate(memory)
            if was_training:
                self.train()
            outputs["seg_logits"] = self.tokens_to_seg_logits(gen_tokens, seq_len)
            outputs["tokens"] = gen_tokens
        elif "seg_logits" not in outputs and not return_loss:
            # Inference without labels: always AR decode.
            was_training = self.training
            self.eval()
            with torch.no_grad():
                gen_tokens = self.generate(memory)
            if was_training:
                self.train()
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
    )
    return model
