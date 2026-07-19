"""Feudal Mamba-MoE Behemoth — full model assembly.

Stacks N FeudalPyramidBlocks and caps them with the *global shared geometric
stratum layer*: a single learned convergence point that sits ABOVE every
pyramid and re-expresses the accumulated hidden stream in one common geometric
basis before the language-model head reads from it.

Geometry contract (see architecture_spec.md):

    tokens [B, S]  (long)
      -> embedding                    [B, S] -> [B, S, D]
      -> N x FeudalPyramidBlock       [B, S, D] -> [B, S, D]   (each residual)
      -> GlobalGeometricStratum       [B, S, D] -> [B, S, D]   (shared basis)
      -> RMSNorm + lm_head            [B, S, D] -> [B, S, V]

The model surfaces the summed Switch load-balance loss across every block's
two minion clusters so the trainer can weight and add it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.feudal_block import FeudalConfig, FeudalPyramidBlock, RMSNorm


@dataclass
class FeudalModelConfig:
    """Geometry of the full stacked model. Wraps the per-block FeudalConfig."""

    vocab_size: int = 32000
    num_blocks: int = 8
    stratum_rank: int = 0  # 0 -> full-rank D; >0 -> low-rank bottleneck for the global stratum
    block: FeudalConfig = field(default_factory=FeudalConfig)

    @property
    def hidden_dim(self) -> int:
        return self.block.hidden_dim

    @classmethod
    def from_json(cls, path: str | Path) -> "FeudalModelConfig":
        raw = json.loads(Path(path).read_text())
        raw.pop("_comment", None)
        block_raw = raw.pop("block", {})
        block_raw.pop("_comment", None)
        return cls(block=FeudalConfig(**block_raw), **raw)


class GlobalGeometricStratum(nn.Module):
    """The global shared geometric stratum layer sitting above all pyramids.

    Every block writes into the same residual stream in its own local basis;
    this module projects that accumulated stream through one shared geometric
    transform (optionally low-rank, D -> r -> D) and mixes it back residually,
    giving a single global coordinate system for the LM head to read.
    """

    def __init__(self, cfg: FeudalModelConfig) -> None:
        super().__init__()
        D, r = cfg.hidden_dim, cfg.stratum_rank
        self.norm = RMSNorm(D, cfg.block.norm_eps)
        if r and r > 0:
            # Low-rank shared basis: D -> r -> D.
            self.to_stratum = nn.Linear(D, r, bias=False)     # [D, r]
            self.from_stratum = nn.Linear(r, D, bias=False)   # [r, D]
            nn.init.normal_(self.to_stratum.weight, std=1.0 / math.sqrt(D))
            nn.init.zeros_(self.from_stratum.weight)          # start as identity (residual only)
        else:
            self.to_stratum = nn.Identity()
            self.from_stratum = nn.Linear(D, D, bias=False)   # [D, D]
            nn.init.zeros_(self.from_stratum.weight)          # start as identity (residual only)
        # Per-channel gate on the stratum contribution, init 0 -> pure identity at start.
        self.stratum_gate = nn.Parameter(torch.zeros(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] -> [B, S, D]
        s = self.from_stratum(self.to_stratum(self.norm(x)))  # [B, S, D]
        return x + self.stratum_gate * s


class FeudalBehemoth(nn.Module):
    """Full Feudal Mamba-MoE model: embedding -> pyramid stack -> global stratum -> head."""

    def __init__(self, cfg: FeudalModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.embedding = nn.Embedding(cfg.vocab_size, D)
        self.blocks = nn.ModuleList(FeudalPyramidBlock(cfg.block) for _ in range(cfg.num_blocks))
        self.global_stratum = GlobalGeometricStratum(cfg)
        self.final_norm = RMSNorm(D, cfg.block.norm_eps)
        self.lm_head = nn.Linear(D, cfg.vocab_size, bias=False)
        # Weight tying: head reads the same basis the embedding writes.
        self.lm_head.weight = self.embedding.weight

    def forward(
        self, tokens: torch.Tensor, return_block_aux: bool = False
    ) -> Dict[str, object]:
        # tokens: [B, S] long -> logits: [B, S, V]
        h = self.embedding(tokens)                       # [B, S, D]

        total_lb = h.new_zeros(())
        block_aux: List[Dict[str, object]] = []
        for block in self.blocks:
            h, aux = block(h)                            # [B, S, D]
            total_lb = total_lb + aux["load_balance_loss"]
            if return_block_aux:
                block_aux.append(aux)

        h = self.global_stratum(h)                       # [B, S, D] shared basis
        h = self.final_norm(h)
        logits = self.lm_head(h)                         # [B, S, V]

        out: Dict[str, object] = {
            "logits": logits,
            "load_balance_loss": total_lb,
        }
        if return_block_aux:
            out["block_aux"] = block_aux
        return out

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def load_minion_fragment(
        self, fragment: Dict[str, torch.Tensor], block_index: int
    ) -> "torch.nn.modules.module._IncompatibleKeys":
        """Load a weight-surgery fragment (from utils.weight_surgery) into one block."""
        if not 0 <= block_index < self.cfg.num_blocks:
            raise IndexError(f"block_index {block_index} out of range [0, {self.cfg.num_blocks})")
        prefixed = {f"blocks.{block_index}.{k}": v for k, v in fragment.items()}
        return self.load_state_dict(prefixed, strict=False)


def build_model(config_path: Optional[str | Path] = None, **overrides) -> FeudalBehemoth:
    """Convenience constructor from a JSON config path or keyword overrides."""
    if config_path is not None:
        cfg = FeudalModelConfig.from_json(config_path)
    else:
        cfg = FeudalModelConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return FeudalBehemoth(cfg)
