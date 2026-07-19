# Feudal Mamba-MoE Behemoth — Phase 1 Architecture Specification

**Status:** Phase 1 — core structural primitives
**Scope:** One `FeudalPyramidBlock` (the repeating unit of the eventual model), the
weight-surgery blueprint for Llama-4-Maverick FFN upcycling, and the geometry
verification suite.

---

## 1. Notation

| Symbol | Meaning | Config field | Base value | Test value |
|--------|---------|--------------|-----------:|-----------:|
| `B` | batch size | — | — | 2–3 |
| `S` | sequence length | — | — | 16–32 |
| `D` | model hidden dim | `hidden_dim` | 5120 | 64 |
| `L` | compressed shared latent dim | `latent_dim` | 1280 | 32 |
| `E` | expansion factor inside each Overlord SSM | `ssm_expand` | 2 | 2 |
| `Di` | Overlord inner dim, `Di = E · L` | derived | 2560 | 64 |
| `H` | SSM heads per Overlord | `ssm_heads` | 40 | 4 |
| `P` | SSM head dim, `P = Di / H` | derived | 64 | 16 |
| `N` | SSM state dim per head | `ssm_state_dim` | 128 | 16 |
| `K` | causal conv kernel width | `ssm_conv_kernel` | 4 | 4 |
| `M` | minion experts **per Overlord cluster** | `minions_per_overlord` | 64 | 4 |
| `k` | active minions per token | `minion_top_k` | 2 | 2 |
| `F` | minion FFN intermediate dim | `minion_ffn_dim` | 2048 | 48 |

Two Overlords exist per block: **Alpha** (macro-logic path) and **Beta**
(structural/syntax path). Their SSM state spaces are *isolated*: no parameter is
shared between them and no tensor crosses between the two paths until the
Convergence Layer.

Total minion count per block is `2·M` (Alpha cluster + Beta cluster). With the
base config this is 128 experts per block, deliberately mirroring the
Llama-4-Maverick-17B-128E expert count so upcycled FFN weights map 1:1 onto
minion slots.

---

## 2. Block-level tensor geometry map

```
                         x  [B, S, D]
                          │
          ┌───────────────┴────────────────┐  (residual copy)
          │                                │
   SharedLatentProjection                  │
   RMSNorm(D) → W_down [D, L]              │
          │  u  [B, S, L]                  │
   ┌──────┴──────┐                         │
   │             │      (same u fed to both paths — the projection is SHARED)
   ▼             ▼                         │
Overlord α    Overlord β                   │
Mamba-2 SSM   Mamba-2 SSM                  │
[B,S,L]→[B,S,L]  [B,S,L]→[B,S,L]           │
   │             │                         │
   ▼             ▼                         │
Minion gate α  Minion gate β               │
top-k over M   top-k over M                │
experts        experts                     │
[B,S,L]→[B,S,L]  [B,S,L]→[B,S,L]           │
   │             │                         │
   └──────┬──────┘                         │
          ▼                                │
   ConvergenceLayer                        │
   concat → [B, S, 2L]                     │
   W_fuse [2L, D] (Cross-Level             │
   Structural Fusion Matrix)               │
          │  y  [B, S, D]                  │
          └────────────► (+) ◄─────────────┘
                          │
                     out [B, S, D]
```

Mathematical constraints enforced by construction (and asserted in
`tests/test_geometry.py`):

1. `Di = ssm_expand · latent_dim` must be divisible by `ssm_heads`.
2. `minion_top_k ≤ minions_per_overlord`.
3. The block is dimension-preserving: `out.shape == x.shape == [B, S, D]`.
4. Strict causality: `out[:, t]` is a function of `x[:, :t+1]` only (the SSM
   scan is a left-to-right recurrence and the conv is left-padded).
5. State isolation: `∂ loss_α / ∂ θ_β = 0` when the loss is taken over the
   Alpha path alone (verified via gradient masking tests).

---

## 3. Shared Latent Projection Layer

```
x  [B, S, D]
   → RMSNorm(D)                # scale-invariant pre-norm
   → Linear(D → L, bias=False) # W_down: [D, L]
u  [B, S, L]
```

The projection is intentionally *shared* between the two Overlords: it defines
one common compressed coordinate system (the "shared geometric stratum" seen
from below), while everything downstream of it is duplicated and isolated.

---

## 4. Overlord layers — isolated Mamba-2 state spaces

Each Overlord is an independent Mamba-2-style selective state-space mixer over
the latent stream. Per-layer geometry:

```
u [B, S, L]
  in_proj: Linear(L → 2·Di + 2·N + H, bias=False)
     split → z  [B, S, Di]      (output gate branch)
             xc [B, S, Di]      (SSM input branch)
             Bc [B, S, N]       (input-dependent B, n_groups = 1)
             Cc [B, S, N]       (input-dependent C, n_groups = 1)
             dt [B, S, H]       (per-head timestep logits)
  causal depthwise Conv1d over concat(xc, Bc, Cc):
     conv_dim = Di + 2N, kernel K, left-pad K-1   # [B, conv_dim, S] → [B, conv_dim, S]
  Δ = softplus(dt + dt_bias)                      # [B, S, H], Δ > 0
  A = −exp(A_log)                                 # [H], scalar-per-head (Mamba-2 SSD form)
  reshape xc → [B, S, H, P]
```

Recurrence (discretized, zero initial state `h₋₁ = 0`):

```
h_t = exp(Δ_t · A) ⊙ h_{t−1} + Δ_t · (x_t ⊗ B_t)     h_t : [B, H, P, N]
y_t = ⟨h_t, C_t⟩_N + D_skip ⊙ x_t                    y_t : [B, H, P]
```

Output path:

```
y  [B, S, Di] = flatten(y_t over heads)
y ← RMSNorm(Di)(y) ⊙ silu(z)        # gated normalization
out_proj: Linear(Di → L, bias=False)
→ [B, S, L], added residually to u
```

Stability constraints: `A < 0` always (parameterized as `−exp(A_log)`), and
`Δ > 0` always (softplus), so `exp(Δ·A) ∈ (0, 1)` — the recurrence is
contractive and cannot blow up over long sequences.

Alpha and Beta share this *shape* but nothing else: separate `in_proj`,
`A_log`, `dt_bias`, conv kernels, norms, and `out_proj`. Isolation is
structural, not conventional.

---

## 5. Minion Dispatch Gate (per Overlord)

Each Overlord owns a private cluster of `M` minion experts. A minion is a
SwiGLU FFN whose parameter names deliberately mirror the Llama FFN layout so
weight surgery is a pure copy:

```
MinionExpert:
  gate_proj: Linear(L → F, bias=False)   # W_gate [L, F]
  up_proj:   Linear(L → F, bias=False)   # W_up   [L, F]
  down_proj: Linear(F → L, bias=False)   # W_down [F, L]
  forward: down_proj( silu(gate_proj(v)) ⊙ up_proj(v) )
```

Routing (Switch/Top-k, computed in float32 for stability):

```
v [B, S, L] → router: Linear(L → M, bias=False) → logits [B, S, M]
p = softmax(logits)                       # [B, S, M]
(w, idx) = top-k(p, k); w ← w / Σw        # renormalized combine weights
out[b,s] = Σ_{j<k} w[b,s,j] · Expert_{idx[b,s,j]}(v[b,s])
```

Auxiliary load-balancing loss per cluster (Switch Transformer form):

```
f_e = fraction of tokens whose top-k set contains e
p̄_e = mean router probability of e
L_aux = M · Σ_e f_e · p̄_e          (minimum M·(1/M)·(1/M)·M = 1 at uniform)
```

The block returns `aux["load_balance_loss"] = L_aux(α) + L_aux(β)` for the
trainer to weight and add.

---

## 6. Top-Level Convergence Layer

```
a [B, S, L]  (alpha path)     b [B, S, L]  (beta path)
a ← RMSNorm(L)(a) ⊙ g_α       b ← RMSNorm(L)(b) ⊙ g_β     # learned per-channel path gates, init 1
c = concat(a, b, dim=-1)                                   # [B, S, 2L]
y = c · W_fuse                                             # W_fuse: [2L, D] — Cross-Level
                                                           # Structural Fusion Matrix
out = x + y                                                # residual to block input
```

`W_fuse` is the single place where the two isolated strata meet; its top `L`
rows read the Alpha stratum and its bottom `L` rows read the Beta stratum, so
post-hoc analysis of cross-stratum mixing reduces to inspecting one matrix.
It is initialized with variance scaled by `1/(2L)` and the residual makes the
block near-identity at init.

---

## 7. Weight surgery contract (`utils/weight_surgery.py`)

Source: Llama-4-Maverick-17B-128E safetensors checkpoints. Two naming layouts
are supported:

* **Fused** (HF Llama-4): `...layers.{i}.feed_forward.experts.gate_up_proj`
  `[n_exp, hidden, 2·ffn]` and `...experts.down_proj` `[n_exp, ffn, hidden]`.
* **Per-expert**: `...experts.{e}.gate_proj.weight` `[ffn, hidden]`, plus
  `up_proj.weight` `[ffn, hidden]` and `down_proj.weight` `[hidden, ffn]`.

Geometry adaptation is *slicing only* (leading-index truncation of rows and
columns to `[F, L]` / `[L, F]` targets). Surgery never fabricates parameters:
if a target dim exceeds the source dim, it raises. Source experts are assigned
round-robin to (overlord, minion-slot) pairs: even source index → Alpha
cluster, odd → Beta cluster.

---

## 8. Verification obligations (`tests/test_geometry.py`)

1. Forward shape preservation for multiple `(B, S)` combinations.
2. Full backward pass: every trainable parameter receives a finite gradient.
3. State-space isolation: gradients from an Alpha-only loss never touch Beta
   parameters, and vice versa.
4. Causality: perturbing `x[:, t:]` leaves `out[:, :t]` bit-identical.
5. Router sanity: probabilities sum to 1; load-balance loss finite and ≥ 1.
6. Weight surgery round-trip: synthetic checkpoint → surgery → `load_state_dict`
   into the minions with exact value equality on sliced regions.
