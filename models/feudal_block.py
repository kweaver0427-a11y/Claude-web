"""Feudal Pyramid Block — core structural primitive of the Feudal Mamba-MoE Behemoth.

Geometry contract (see architecture_spec.md for the full derivation):

    x [B, S, D]
      -> SharedLatentProjection            [B, S, D] -> [B, S, L]
      -> Overlord alpha (Mamba-2 SSM)      [B, S, L] -> [B, S, L]   (isolated state space)
      -> Overlord beta  (Mamba-2 SSM)      [B, S, L] -> [B, S, L]   (isolated state space)
      -> MinionDispatchGate per overlord   [B, S, L] -> [B, S, L]   (top-k over M experts)
      -> ConvergenceLayer                  2 x [B, S, L] -> [B, S, D]
      -> residual add                      [B, S, D]

Symbols: B batch, S sequence, D hidden_dim, L latent_dim, Di = ssm_expand * L,
H ssm_heads, P = Di // H, N ssm_state_dim, M minions_per_overlord,
F minion_ffn_dim, k minion_top_k.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FeudalConfig:
    """Static geometry of one FeudalPyramidBlock."""

    hidden_dim: int = 5120          # D
    latent_dim: int = 1280          # L
    ssm_expand: int = 2             # E (Di = E * L)
    ssm_heads: int = 40             # H
    ssm_state_dim: int = 128        # N
    ssm_conv_kernel: int = 4        # K
    minions_per_overlord: int = 64  # M (per cluster; block total is 2*M)
    minion_top_k: int = 2           # k
    minion_ffn_dim: int = 2048      # F
    norm_eps: float = 1e-6
    overlord_names: Tuple[str, str] = field(default=("alpha", "beta"))

    @property
    def ssm_inner_dim(self) -> int:  # Di
        return self.ssm_expand * self.latent_dim

    @property
    def ssm_head_dim(self) -> int:  # P
        return self.ssm_inner_dim // self.ssm_heads

    def __post_init__(self) -> None:
        if self.ssm_inner_dim % self.ssm_heads != 0:
            raise ValueError(
                f"ssm_expand * latent_dim ({self.ssm_inner_dim}) must be divisible "
                f"by ssm_heads ({self.ssm_heads})"
            )
        if self.minion_top_k > self.minions_per_overlord:
            raise ValueError(
                f"minion_top_k ({self.minion_top_k}) cannot exceed "
                f"minions_per_overlord ({self.minions_per_overlord})"
            )
        if self.ssm_conv_kernel < 1:
            raise ValueError("ssm_conv_kernel must be >= 1")


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [..., dim] -> [..., dim]
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SharedLatentProjection(nn.Module):
    """Down-projects the hidden stream into the shared compressed latent space.

    Both Overlords read the SAME latent stream; this module defines the common
    coordinate system beneath the isolated strata.
    """

    def __init__(self, cfg: FeudalConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.down = nn.Linear(cfg.hidden_dim, cfg.latent_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] -> normed: [B, S, D] -> latent: [B, S, L]
        return self.down(self.norm(x))


class Mamba2StateSpaceLayer(nn.Module):
    """One isolated Overlord: a Mamba-2-style selective SSM over the latent stream.

    Scalar-per-head A (SSD form), input-dependent (dt, B, C), causal depthwise
    conv over the (x, B, C) channels, gated RMSNorm output, residual add.
    Pure PyTorch sequential scan — numerically faithful, hardware-agnostic.
    """

    def __init__(self, cfg: FeudalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        L, Di, H, N, K = (
            cfg.latent_dim,
            cfg.ssm_inner_dim,
            cfg.ssm_heads,
            cfg.ssm_state_dim,
            cfg.ssm_conv_kernel,
        )
        self.pre_norm = RMSNorm(L, cfg.norm_eps)
        # Fused input projection -> [z (Di), x (Di), B (N), C (N), dt (H)]
        self.in_proj = nn.Linear(L, 2 * Di + 2 * N + H, bias=False)

        # Causal depthwise conv over the concatenated (x, B, C) channels.
        conv_dim = Di + 2 * N
        self.conv = nn.Conv1d(
            conv_dim, conv_dim, kernel_size=K, groups=conv_dim, padding=0, bias=True
        )

        # dt bias initialized so softplus(dt_bias) spans [1e-3, 1e-1] log-uniformly.
        dt_init = torch.exp(
            torch.rand(H) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3)
        )
        inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(inv_softplus)

        # A = -exp(A_log) < 0 always: contractive recurrence.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        # Per-head skip connection through the SSM.
        self.D_skip = nn.Parameter(torch.ones(H))

        self.gate_norm = RMSNorm(Di, cfg.norm_eps)
        self.out_proj = nn.Linear(Di, L, bias=False)

    def _ssm_scan(
        self,
        xh: torch.Tensor,   # [B, S, H, P]  SSM input, split into heads
        dt: torch.Tensor,   # [B, S, H]     positive timesteps
        Bc: torch.Tensor,   # [B, S, N]     input-dependent input matrix
        Cc: torch.Tensor,   # [B, S, N]     input-dependent output matrix
    ) -> torch.Tensor:
        B_, S, H, P = xh.shape
        N = Bc.shape[-1]
        A = -torch.exp(self.A_log.float())               # [H]
        h = xh.new_zeros(B_, H, P, N)                    # [B, H, P, N] isolated state
        ys = []
        for t in range(S):
            dt_t = dt[:, t]                              # [B, H]
            dA = torch.exp(dt_t * A)                     # [B, H] in (0, 1)
            dBx = (dt_t.unsqueeze(-1) * xh[:, t]).unsqueeze(-1) * Bc[:, t][
                :, None, None, :
            ]                                            # [B, H, P, N]
            h = h * dA[:, :, None, None] + dBx           # [B, H, P, N]
            y_t = (h * Cc[:, t][:, None, None, :]).sum(-1)  # [B, H, P]
            y_t = y_t + self.D_skip[None, :, None] * xh[:, t]
            ys.append(y_t)
        return torch.stack(ys, dim=1)                    # [B, S, H, P]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, S, L] -> out: [B, S, L]
        cfg = self.cfg
        Di, H, N, K = cfg.ssm_inner_dim, cfg.ssm_heads, cfg.ssm_state_dim, cfg.ssm_conv_kernel
        B_, S, _ = u.shape

        residual = u
        zxbcdt = self.in_proj(self.pre_norm(u))          # [B, S, 2*Di + 2*N + H]
        z, xc, Bc, Cc, dt = torch.split(zxbcdt, [Di, Di, N, N, H], dim=-1)

        # Causal depthwise conv over (x, B, C): left-pad K-1 so position t sees <= t.
        xbc = torch.cat([xc, Bc, Cc], dim=-1)            # [B, S, Di + 2N]
        xbc = F.pad(xbc.transpose(1, 2), (K - 1, 0))     # [B, Di + 2N, S + K - 1]
        xbc = F.silu(self.conv(xbc)).transpose(1, 2)     # [B, S, Di + 2N]
        xc, Bc, Cc = torch.split(xbc, [Di, N, N], dim=-1)

        dt = F.softplus(dt + self.dt_bias)               # [B, S, H], strictly > 0
        xh = xc.reshape(B_, S, H, cfg.ssm_head_dim)      # [B, S, H, P]
        y = self._ssm_scan(xh, dt, Bc, Cc)               # [B, S, H, P]
        y = y.reshape(B_, S, Di)                         # [B, S, Di]

        y = self.gate_norm(y) * F.silu(z)                # gated output, [B, S, Di]
        return residual + self.out_proj(y)               # [B, S, L]


class MinionExpert(nn.Module):
    """One minion: a SwiGLU FFN with Llama-compatible parameter naming.

    gate_proj / up_proj: [B*, L] -> [B*, F];  down_proj: [B*, F] -> [B*, L].
    Names match Llama FFN modules so weight surgery is a pure tensor copy.
    """

    def __init__(self, cfg: FeudalConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.latent_dim, cfg.minion_ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.latent_dim, cfg.minion_ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.minion_ffn_dim, cfg.latent_dim, bias=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: [T, L] -> [T, L]  (T = number of tokens routed to this expert)
        return self.down_proj(F.silu(self.gate_proj(v)) * self.up_proj(v))


class MinionDispatchGate(nn.Module):
    """Routes each latent token to the top-k minions of one Overlord's cluster.

    Router probabilities are computed in float32; combine weights are the
    renormalized top-k softmax mass. Returns the mixed expert output plus a
    Switch-style load-balancing loss for the trainer.
    """

    def __init__(self, cfg: FeudalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.latent_dim, cfg.minions_per_overlord, bias=False)
        self.experts = nn.ModuleList(
            MinionExpert(cfg) for _ in range(cfg.minions_per_overlord)
        )

    def forward(self, v: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # v: [B, S, L] -> out: [B, S, L]
        cfg = self.cfg
        B_, S, L = v.shape
        M, k = cfg.minions_per_overlord, cfg.minion_top_k

        flat = v.reshape(B_ * S, L)                      # [T, L], T = B*S
        logits = self.router(flat).float()               # [T, M]
        probs = logits.softmax(dim=-1)                   # [T, M]
        topk_w, topk_idx = probs.topk(k, dim=-1)         # [T, k], [T, k]
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)                     # [T, L]
        for e, expert in enumerate(self.experts):
            token_mask, slot = (topk_idx == e).max(dim=-1)   # [T] bool, [T] slot idx
            if not token_mask.any():
                continue
            rows = token_mask.nonzero(as_tuple=True)[0]      # [T_e]
            w = topk_w[rows, slot[rows]].to(v.dtype)         # [T_e]
            out = out.index_add(0, rows, w.unsqueeze(-1) * expert(flat[rows]))

        # Switch load-balance loss: M * sum_e f_e * pbar_e  (== 1 at uniform routing).
        with torch.no_grad():
            hits = torch.zeros_like(probs)               # [T, M]
            hits.scatter_(1, topk_idx, 1.0)
            f = hits.mean(dim=0)                         # [M] token fraction per expert
        pbar = probs.mean(dim=0)                         # [M] mean router prob
        aux = {
            "load_balance_loss": M * (f * pbar).sum(),
            "router_probs": probs.reshape(B_, S, M),
            "expert_token_fraction": f,
        }
        return out.reshape(B_, S, L), aux


class ConvergenceLayer(nn.Module):
    """Fuses the isolated Alpha/Beta strata back into the hidden stream.

    The Cross-Level Structural Fusion Matrix W_fuse [2L, D] is the single
    point of contact between the two state spaces: rows [0, L) read the Alpha
    stratum, rows [L, 2L) read the Beta stratum.
    """

    def __init__(self, cfg: FeudalConfig) -> None:
        super().__init__()
        L, D = cfg.latent_dim, cfg.hidden_dim
        self.alpha_norm = RMSNorm(L, cfg.norm_eps)
        self.beta_norm = RMSNorm(L, cfg.norm_eps)
        self.alpha_gate = nn.Parameter(torch.ones(L))    # per-channel path gate
        self.beta_gate = nn.Parameter(torch.ones(L))
        self.fusion = nn.Linear(2 * L, D, bias=False)    # W_fuse: [2L, D]
        nn.init.normal_(self.fusion.weight, std=1.0 / math.sqrt(2 * L))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: [B, S, L] -> fused: [B, S, D]
        a = self.alpha_norm(a) * self.alpha_gate         # [B, S, L]
        b = self.beta_norm(b) * self.beta_gate           # [B, S, L]
        return self.fusion(torch.cat([a, b], dim=-1))    # [B, S, 2L] -> [B, S, D]


class FeudalPyramidBlock(nn.Module):
    """The full pyramid: shared stratum -> isolated Overlords -> minion clusters -> convergence.

    forward: x [B, S, D] -> (out [B, S, D], aux dict).
    aux["load_balance_loss"] is the summed Switch loss over both clusters;
    per-cluster router telemetry lives under aux["alpha"] / aux["beta"].
    """

    def __init__(self, cfg: FeudalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.shared_projection = SharedLatentProjection(cfg)
        self.overlord_alpha = Mamba2StateSpaceLayer(cfg)   # macro-logic stratum
        self.overlord_beta = Mamba2StateSpaceLayer(cfg)    # structural/syntax stratum
        self.minions_alpha = MinionDispatchGate(cfg)
        self.minions_beta = MinionDispatchGate(cfg)
        self.convergence = ConvergenceLayer(cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
        # x: [B, S, D] -> out: [B, S, D]
        u = self.shared_projection(x)                    # [B, S, D] -> [B, S, L]

        a = self.overlord_alpha(u)                       # [B, S, L] (isolated space)
        b = self.overlord_beta(u)                        # [B, S, L] (isolated space)

        a_moe, aux_a = self.minions_alpha(a)             # [B, S, L]
        b_moe, aux_b = self.minions_beta(b)              # [B, S, L]
        a = a + a_moe                                    # residual within stratum
        b = b + b_moe

        y = self.convergence(a, b)                       # 2 x [B, S, L] -> [B, S, D]
        out = x + y                                      # [B, S, D]

        aux: Dict[str, object] = {
            "load_balance_loss": aux_a["load_balance_loss"] + aux_b["load_balance_loss"],
            "alpha": aux_a,
            "beta": aux_b,
        }
        return out, aux
