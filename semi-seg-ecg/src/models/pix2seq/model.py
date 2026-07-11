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
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = tokenizer.max_seq_len

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
        """Encode ECG (B, 1, T) to memory (B, L, d_model)."""
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
        # Causal mask (PyTorch 1.11-compatible)
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
        )
        return self.lm_head(out)

    @torch.no_grad()
    def generate(self, memory: Tensor, max_len: Optional[int] = None) -> Tensor:
        """Autoregressive decode to token ids (B, S)."""
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

        # Pad to max_seq_len for consistent batch decode
        if tokens.size(1) < self.max_seq_len:
            pad_len = self.max_seq_len - tokens.size(1)
            tokens = F.pad(tokens, (0, pad_len), value=pad)
        else:
            tokens = tokens[:, : self.max_seq_len]
        return tokens

    def tokens_to_seg_logits(self, tokens: Tensor, seq_len: int) -> Tensor:
        """Rasterize tokens to soft one-hot logits (B, num_classes, T)."""
        masks = self.tokenizer.batch_decode(tokens)  # (B, T)
        if masks.size(1) != seq_len:
            # Should match signal_length; crop/pad if needed
            if masks.size(1) > seq_len:
                masks = masks[:, :seq_len]
            else:
                masks = F.pad(masks, (0, seq_len - masks.size(1)), value=0)
        # Hard one-hot as logits (large margin) for MeanIoU path that softmaxes
        num_classes = 4
        logits = F.one_hot(masks.clamp(0, num_classes - 1), num_classes=num_classes)
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
            decode: if True, autoregressive decode for seg_logits (eval);
                    if False with labels, use teacher-forced token argmax
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
            if not decode:
                pred_tokens = torch.cat(
                    [target_tokens[:, :1], logits.argmax(dim=-1)],
                    dim=1,
                )
                outputs["seg_logits"] = self.tokens_to_seg_logits(pred_tokens, seq_len)

        if decode or "seg_logits" not in outputs:
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
    tokenizer = SegmentTokenizer(
        signal_length=config["dataset"].get("signal_length", 2500),
        num_bins=p2s.get("num_bins", 250),
        max_segments=p2s.get("max_segments", 32),
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
    )
    return model
