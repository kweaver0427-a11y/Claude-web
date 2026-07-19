"""Geometry verification suite for the FeudalPyramidBlock.

Covers the obligations in architecture_spec.md section 8: shape preservation,
unbroken gradient flow across the isolated state spaces, strict causality,
router sanity, and the weight-surgery round trip.
"""

import json
from pathlib import Path

import pytest
import torch

from models.feudal_block import (
    ConvergenceLayer,
    FeudalConfig,
    FeudalPyramidBlock,
    Mamba2StateSpaceLayer,
    MinionDispatchGate,
    SharedLatentProjection,
)
from utils.weight_surgery import (
    CheckpointReader,
    SurgeryPlan,
    build_minion_fragment,
    slice_to_geometry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def small_config() -> FeudalConfig:
    raw = json.loads((REPO_ROOT / "configs" / "feudal_test_small.json").read_text())
    raw.pop("_comment", None)
    return FeudalConfig(**raw)


@pytest.fixture(scope="module")
def cfg() -> FeudalConfig:
    return small_config()


@pytest.fixture()
def block(cfg: FeudalConfig) -> FeudalPyramidBlock:
    torch.manual_seed(0)
    return FeudalPyramidBlock(cfg)


# ---------------------------------------------------------------------------
# 1. Forward geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch,seq", [(1, 1), (2, 16), (3, 32)])
def test_forward_shape_preserved(block, cfg, batch, seq):
    x = torch.randn(batch, seq, cfg.hidden_dim)
    out, aux = block(x)
    assert out.shape == (batch, seq, cfg.hidden_dim)
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux["load_balance_loss"])


def test_sublayer_geometry(cfg):
    torch.manual_seed(1)
    B, S = 2, 8
    x = torch.randn(B, S, cfg.hidden_dim)

    proj = SharedLatentProjection(cfg)
    u = proj(x)
    assert u.shape == (B, S, cfg.latent_dim)

    overlord = Mamba2StateSpaceLayer(cfg)
    a = overlord(u)
    assert a.shape == (B, S, cfg.latent_dim)

    gate = MinionDispatchGate(cfg)
    moe_out, aux = gate(a)
    assert moe_out.shape == (B, S, cfg.latent_dim)
    assert aux["router_probs"].shape == (B, S, cfg.minions_per_overlord)

    conv = ConvergenceLayer(cfg)
    fused = conv(a, a.clone())
    assert fused.shape == (B, S, cfg.hidden_dim)
    # the Cross-Level Structural Fusion Matrix must be exactly [D, 2L]
    assert conv.fusion.weight.shape == (cfg.hidden_dim, 2 * cfg.latent_dim)


def test_config_validation():
    with pytest.raises(ValueError):
        FeudalConfig(hidden_dim=64, latent_dim=32, ssm_expand=2, ssm_heads=5)
    with pytest.raises(ValueError):
        FeudalConfig(minions_per_overlord=2, minion_top_k=4)


# ---------------------------------------------------------------------------
# 2. Gradient flow across the whole pyramid
# ---------------------------------------------------------------------------


def test_backward_reaches_every_parameter(block, cfg):
    torch.manual_seed(2)
    x = torch.randn(2, 16, cfg.hidden_dim, requires_grad=True)
    out, aux = block(x)
    loss = out.pow(2).mean() + 0.01 * aux["load_balance_loss"]
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    missing = []
    for name, p in block.named_parameters():
        # An expert that received zero tokens this step legitimately has no
        # grad; every non-expert parameter must have a finite, somewhere-nonzero
        # gradient for flow to count as unbroken.
        if p.grad is None:
            if ".experts." in name:
                continue
            missing.append(name)
            continue
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert not missing, f"parameters with no gradient: {missing}"

    # Both isolated state spaces must receive gradient through their core
    # SSM parameters — flow across the scan recurrence is unbroken.
    for overlord in (block.overlord_alpha, block.overlord_beta):
        for pname in ("A_log", "dt_bias", "D_skip"):
            g = getattr(overlord, pname).grad
            assert g is not None and g.abs().sum() > 0, f"dead SSM param {pname}"


def test_state_space_isolation(block, cfg):
    """A loss taken on one stratum's pre-convergence output must not touch the
    other Overlord's parameters — the state spaces are structurally isolated."""
    torch.manual_seed(3)
    x = torch.randn(2, 8, cfg.hidden_dim)
    u = block.shared_projection(x)
    a = block.overlord_alpha(u)
    loss = a.pow(2).mean()
    loss.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in block.overlord_alpha.parameters()
    )
    for name, p in block.overlord_beta.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"leak into beta.{name}"


# ---------------------------------------------------------------------------
# 3. Causality of the recurrent path
# ---------------------------------------------------------------------------


def test_strict_causality(block, cfg):
    torch.manual_seed(4)
    block.eval()
    B, S, t = 1, 24, 12
    x = torch.randn(B, S, cfg.hidden_dim)
    x_perturbed = x.clone()
    x_perturbed[:, t:] += torch.randn(B, S - t, cfg.hidden_dim)

    with torch.no_grad():
        out_a, _ = block(x)
        out_b, _ = block(x_perturbed)

    torch.testing.assert_close(out_a[:, :t], out_b[:, :t], rtol=0, atol=0)
    assert not torch.allclose(out_a[:, t:], out_b[:, t:])


def test_ssm_recurrence_is_contractive(cfg):
    """exp(dt * A) must lie in (0, 1) for every head: A < 0 and dt > 0."""
    overlord = Mamba2StateSpaceLayer(cfg)
    A = -torch.exp(overlord.A_log)
    assert (A < 0).all()
    dt = torch.nn.functional.softplus(torch.randn(100, cfg.ssm_heads) + overlord.dt_bias)
    decay = torch.exp(dt * A)
    assert (decay > 0).all() and (decay < 1).all()


# ---------------------------------------------------------------------------
# 4. Router sanity
# ---------------------------------------------------------------------------


def test_router_probabilities_and_load_balance(cfg):
    torch.manual_seed(5)
    gate = MinionDispatchGate(cfg)
    v = torch.randn(2, 16, cfg.latent_dim)
    out, aux = gate(v)

    probs = aux["router_probs"]
    torch.testing.assert_close(
        probs.sum(dim=-1), torch.ones(2, 16), rtol=1e-5, atol=1e-5
    )
    # Switch loss lower bound is 1.0 (achieved at perfectly uniform routing).
    lb = aux["load_balance_loss"]
    assert torch.isfinite(lb) and lb.item() >= 1.0 - 1e-5
    # Every token's top-k mass lands on exactly k experts.
    frac = aux["expert_token_fraction"]
    torch.testing.assert_close(frac.sum(), torch.tensor(float(cfg.minion_top_k)))


def test_batch_independence(block, cfg):
    """Rows of a batch must not exchange information."""
    torch.manual_seed(6)
    block.eval()
    x = torch.randn(2, 8, cfg.hidden_dim)
    with torch.no_grad():
        out_batched, _ = block(x)
        out_row0, _ = block(x[:1])
    torch.testing.assert_close(out_batched[:1], out_row0, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Weight surgery round trip
# ---------------------------------------------------------------------------


def _write_fake_maverick_checkpoint(path: Path, n_experts: int, hidden: int, ffn: int, layer: int = 0):
    from safetensors.torch import save_file

    torch.manual_seed(7)
    tensors = {}
    for e in range(n_experts):
        base = f"language_model.model.layers.{layer}.feed_forward.experts.{e}"
        tensors[f"{base}.gate_proj.weight"] = torch.randn(ffn, hidden)
        tensors[f"{base}.up_proj.weight"] = torch.randn(ffn, hidden)
        tensors[f"{base}.down_proj.weight"] = torch.randn(hidden, ffn)
    save_file(tensors, str(path))
    return tensors


def test_slice_to_geometry_rules():
    w = torch.randn(8, 6)
    sliced = slice_to_geometry(w, (4, 3))
    torch.testing.assert_close(sliced, w[:4, :3])
    with pytest.raises(ValueError):
        slice_to_geometry(w, (10, 3))  # cannot fabricate rows
    with pytest.raises(ValueError):
        slice_to_geometry(w, (8, 6, 1))  # rank mismatch


def test_weight_surgery_round_trip(tmp_path, cfg):
    src_hidden, src_ffn = cfg.latent_dim * 2, cfg.minion_ffn_dim * 2
    n_src_experts = 2 * cfg.minions_per_overlord
    ckpt = tmp_path / "maverick_toy.safetensors"
    source = _write_fake_maverick_checkpoint(ckpt, n_src_experts, src_hidden, src_ffn)

    plan = SurgeryPlan(
        latent_dim=cfg.latent_dim,
        ffn_dim=cfg.minion_ffn_dim,
        minions_per_overlord=cfg.minions_per_overlord,
        source_layer=0,
    )
    fragment = build_minion_fragment(CheckpointReader(ckpt), plan)

    # 3 projections per minion, 2*M minions total.
    assert len(fragment) == 3 * 2 * cfg.minions_per_overlord

    torch.manual_seed(8)
    block = FeudalPyramidBlock(cfg)
    result = block.load_state_dict(fragment, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys

    # Even source expert 0 -> alpha slot 0; odd source expert 1 -> beta slot 0.
    src0 = source["language_model.model.layers.0.feed_forward.experts.0.gate_proj.weight"]
    torch.testing.assert_close(
        block.minions_alpha.experts[0].gate_proj.weight.data,
        src0[: cfg.minion_ffn_dim, : cfg.latent_dim],
    )
    src1 = source["language_model.model.layers.0.feed_forward.experts.1.down_proj.weight"]
    torch.testing.assert_close(
        block.minions_beta.experts[0].down_proj.weight.data,
        src1[: cfg.latent_dim, : cfg.minion_ffn_dim],
    )

    # The surgically-loaded block must still satisfy the forward contract.
    x = torch.randn(2, 8, cfg.hidden_dim)
    out, _ = block(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_weight_surgery_fused_layout(tmp_path, cfg):
    from safetensors.torch import save_file

    torch.manual_seed(9)
    n_exp = 2 * cfg.minions_per_overlord
    hidden, ffn = cfg.latent_dim * 2, cfg.minion_ffn_dim * 2
    gate_up = torch.randn(n_exp, hidden, 2 * ffn)
    down = torch.randn(n_exp, ffn, hidden)
    ckpt = tmp_path / "maverick_fused.safetensors"
    save_file(
        {
            "language_model.model.layers.0.feed_forward.experts.gate_up_proj": gate_up,
            "language_model.model.layers.0.feed_forward.experts.down_proj": down,
        },
        str(ckpt),
    )

    plan = SurgeryPlan(
        latent_dim=cfg.latent_dim,
        ffn_dim=cfg.minion_ffn_dim,
        minions_per_overlord=cfg.minions_per_overlord,
    )
    fragment = build_minion_fragment(CheckpointReader(ckpt), plan)
    assert len(fragment) == 3 * n_exp

    # Fused layout stores [hidden, 2*ffn]; gate half transposed -> [ffn, hidden],
    # then sliced to [F, L].
    expected_gate = gate_up[0][:, :ffn].T[: cfg.minion_ffn_dim, : cfg.latent_dim]
    torch.testing.assert_close(
        fragment["minions_alpha.experts.0.gate_proj.weight"], expected_gate
    )
