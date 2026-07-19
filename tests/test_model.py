"""Verification suite for the full stacked FeudalBehemoth model.

Covers: end-to-end forward geometry, gradient flow through the whole stack and
the global shared geometric stratum layer, aggregation of per-block load-balance
losses, causality of the stacked recurrence, the near-identity-at-init property
of the global stratum, and surgery-fragment loading into a chosen block.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.feudal_model import (
    FeudalBehemoth,
    FeudalModelConfig,
    GlobalGeometricStratum,
    build_model,
)
from utils.weight_surgery import CheckpointReader, SurgeryPlan, build_minion_fragment

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "configs" / "feudal_model_test_small.json"


@pytest.fixture(scope="module")
def cfg() -> FeudalModelConfig:
    return FeudalModelConfig.from_json(CONFIG)


@pytest.fixture()
def model(cfg: FeudalModelConfig) -> FeudalBehemoth:
    torch.manual_seed(0)
    return FeudalBehemoth(cfg)


# ---------------------------------------------------------------------------
# Forward geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch,seq", [(1, 1), (2, 12), (3, 20)])
def test_forward_logits_shape(model, cfg, batch, seq):
    tokens = torch.randint(0, cfg.vocab_size, (batch, seq))
    out = model(tokens)
    assert out["logits"].shape == (batch, seq, cfg.vocab_size)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["load_balance_loss"])


def test_load_balance_aggregates_over_blocks(model, cfg):
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(tokens, return_block_aux=True)
    assert len(out["block_aux"]) == cfg.num_blocks
    manual = sum(a["load_balance_loss"] for a in out["block_aux"])
    torch.testing.assert_close(out["load_balance_loss"], manual)
    # Each block's Switch loss respects the >= 1 lower bound, so the sum does too.
    assert out["load_balance_loss"].item() >= cfg.num_blocks - 1e-4


def test_weight_tying(model):
    assert model.lm_head.weight is model.embedding.weight


def test_build_model_helper_and_overrides():
    m = build_model(CONFIG, num_blocks=2)
    assert len(m.blocks) == 2
    assert m.num_parameters() > 0


# ---------------------------------------------------------------------------
# Gradient flow through the full stack + global stratum
# ---------------------------------------------------------------------------


def test_backward_full_stack(model, cfg):
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(tokens)
    targets = torch.randint(0, cfg.vocab_size, (2, 12))
    loss = F.cross_entropy(
        out["logits"].reshape(-1, cfg.vocab_size), targets.reshape(-1)
    ) + 0.01 * out["load_balance_loss"]
    loss.backward()

    missing = []
    for name, p in model.named_parameters():
        if p.grad is None:
            # Idle experts (no routed tokens) legitimately have no grad.
            if ".experts." in name:
                continue
            missing.append(name)
            continue
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert not missing, f"parameters with no gradient: {missing}"

    # The global stratum must participate: its gate starts at 0 but the loss
    # depends on it once the head reads a non-degenerate stream, so its
    # projection weights must receive gradient.
    assert model.global_stratum.from_stratum.weight.grad is not None
    assert model.global_stratum.stratum_gate.grad is not None


def test_every_block_receives_gradient(model, cfg):
    torch.manual_seed(2)
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(tokens)
    out["logits"].pow(2).mean().backward()
    for i, block in enumerate(model.blocks):
        g = block.overlord_alpha.A_log.grad
        assert g is not None and g.abs().sum() > 0, f"block {i} alpha SSM got no grad"


# ---------------------------------------------------------------------------
# Global geometric stratum properties
# ---------------------------------------------------------------------------


def test_global_stratum_identity_at_init(cfg):
    """stratum_gate init 0 (and zeroed from_stratum) => exact identity at init."""
    torch.manual_seed(3)
    stratum = GlobalGeometricStratum(cfg)
    x = torch.randn(2, 8, cfg.hidden_dim)
    torch.testing.assert_close(stratum(x), x, rtol=0, atol=0)


def test_global_stratum_low_rank_shape(cfg):
    stratum = GlobalGeometricStratum(cfg)
    # Config uses stratum_rank=16 < D=64, so the bottleneck is genuinely low-rank.
    assert cfg.stratum_rank > 0
    assert stratum.to_stratum.weight.shape == (cfg.stratum_rank, cfg.hidden_dim)
    assert stratum.from_stratum.weight.shape == (cfg.hidden_dim, cfg.stratum_rank)


def test_full_rank_stratum_variant():
    cfg = FeudalModelConfig.from_json(CONFIG)
    cfg.stratum_rank = 0
    stratum = GlobalGeometricStratum(cfg)
    x = torch.randn(1, 4, cfg.hidden_dim)
    # Full-rank D->D path, still identity at init.
    torch.testing.assert_close(stratum(x), x, rtol=0, atol=0)
    assert stratum.from_stratum.weight.shape == (cfg.hidden_dim, cfg.hidden_dim)


# ---------------------------------------------------------------------------
# Causality of the stacked recurrence
# ---------------------------------------------------------------------------


def test_stacked_causality(model, cfg):
    torch.manual_seed(4)
    model.eval()
    B, S, t = 1, 18, 9
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    perturbed = tokens.clone()
    perturbed[:, t:] = torch.randint(0, cfg.vocab_size, (B, S - t))

    with torch.no_grad():
        a = model(tokens)["logits"]
        b = model(perturbed)["logits"]

    torch.testing.assert_close(a[:, :t], b[:, :t], rtol=0, atol=0)
    assert not torch.allclose(a[:, t:], b[:, t:])


# ---------------------------------------------------------------------------
# Surgery fragment loading into the stacked model
# ---------------------------------------------------------------------------


def test_load_minion_fragment_into_block(tmp_path, model, cfg):
    from safetensors.torch import save_file

    block_cfg = cfg.block
    n_exp = 2 * block_cfg.minions_per_overlord
    hidden, ffn = block_cfg.latent_dim, block_cfg.minion_ffn_dim
    ckpt = tmp_path / "toy.safetensors"

    torch.manual_seed(5)
    tensors = {}
    for e in range(n_exp):
        base = f"language_model.model.layers.0.feed_forward.experts.{e}"
        tensors[f"{base}.gate_proj.weight"] = torch.randn(ffn, hidden)
        tensors[f"{base}.up_proj.weight"] = torch.randn(ffn, hidden)
        tensors[f"{base}.down_proj.weight"] = torch.randn(hidden, ffn)
    save_file(tensors, str(ckpt))

    plan = SurgeryPlan(
        latent_dim=block_cfg.latent_dim,
        ffn_dim=block_cfg.minion_ffn_dim,
        minions_per_overlord=block_cfg.minions_per_overlord,
    )
    fragment = build_minion_fragment(CheckpointReader(ckpt), plan)

    result = model.load_minion_fragment(fragment, block_index=1)
    assert not result.unexpected_keys, result.unexpected_keys

    # Values landed in block 1's alpha cluster, not block 0.
    src0 = tensors["language_model.model.layers.0.feed_forward.experts.0.gate_proj.weight"]
    torch.testing.assert_close(
        model.blocks[1].minions_alpha.experts[0].gate_proj.weight.data, src0
    )
    assert not torch.allclose(
        model.blocks[0].minions_alpha.experts[0].gate_proj.weight.data, src0
    )

    with pytest.raises(IndexError):
        model.load_minion_fragment(fragment, block_index=99)

    tokens = torch.randint(0, cfg.vocab_size, (1, 6))
    assert torch.isfinite(model(tokens)["logits"]).all()
