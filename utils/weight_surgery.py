"""Weight surgery: upcycle Llama-4-Maverick FFN expert weights into minion slots.

This is the structural blueprint for Phase 2 upcycling. It parses a
Llama-4-Maverick-17B-128E safetensors checkpoint (single-file or sharded),
locates the routed-expert FFN tensors for a chosen decoder layer, slices them
down to the feudal block's minion geometry, and emits a state-dict fragment
that loads directly into ``FeudalPyramidBlock.minions_alpha`` /
``minions_beta``.

Supported source layouts (auto-detected):

* Fused HF Llama-4 layout::

      language_model.model.layers.{i}.feed_forward.experts.gate_up_proj   [n_exp, hidden, 2*ffn]
      language_model.model.layers.{i}.feed_forward.experts.down_proj      [n_exp, ffn, hidden]

* Per-expert layout::

      ...layers.{i}.feed_forward.experts.{e}.gate_proj.weight   [ffn, hidden]
      ...layers.{i}.feed_forward.experts.{e}.up_proj.weight     [ffn, hidden]
      ...layers.{i}.feed_forward.experts.{e}.down_proj.weight   [hidden, ffn]

Geometry adaptation is leading-index slicing only — surgery never fabricates
parameters. Source expert ``e`` maps round-robin: even ``e`` -> Alpha cluster
slot ``e // 2``, odd ``e`` -> Beta cluster slot ``e // 2``.

CLI::

    python -m utils.weight_surgery CHECKPOINT_PATH --layer 0 \
        --latent-dim 1280 --ffn-dim 2048 --minions-per-overlord 64 \
        --out surgery_fragment.pt
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

_EXPERT_PER_TENSOR_RE = re.compile(
    r"layers\.(?P<layer>\d+)\.feed_forward\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_EXPERT_FUSED_RE = re.compile(
    r"layers\.(?P<layer>\d+)\.feed_forward\.experts\."
    r"(?P<proj>gate_up_proj|down_proj)$"
)


@dataclass
class SurgeryPlan:
    """Target minion geometry the sliced weights must land in."""

    latent_dim: int           # L: minion input/output width
    ffn_dim: int              # F: minion intermediate width
    minions_per_overlord: int  # M: slots per cluster (block total 2*M)
    source_layer: int = 0     # which decoder layer of the donor to harvest

    @property
    def total_minions(self) -> int:
        return 2 * self.minions_per_overlord


class CheckpointReader:
    """Uniform tensor access over a single .safetensors file or a sharded dir.

    For a directory, ``model.safetensors.index.json`` (or any ``*.index.json``)
    maps tensor names to shard files; shards are opened lazily and tensors are
    read individually so the full checkpoint never has to fit in memory.
    """

    def __init__(self, path: str | Path) -> None:
        from safetensors import safe_open  # lazy: only needed when surgery runs

        self._safe_open = safe_open
        self.path = Path(path)
        self._name_to_file: Dict[str, Path] = {}

        if self.path.is_file():
            with safe_open(self.path, framework="pt") as f:
                for name in f.keys():
                    self._name_to_file[name] = self.path
        elif self.path.is_dir():
            index_files = sorted(self.path.glob("*.index.json"))
            if index_files:
                index = json.loads(index_files[0].read_text())
                for name, shard in index["weight_map"].items():
                    self._name_to_file[name] = self.path / shard
            else:
                for shard in sorted(self.path.glob("*.safetensors")):
                    with safe_open(shard, framework="pt") as f:
                        for name in f.keys():
                            self._name_to_file[name] = shard
            if not self._name_to_file:
                raise FileNotFoundError(f"no safetensors content under {self.path}")
        else:
            raise FileNotFoundError(str(self.path))

    def names(self) -> List[str]:
        return sorted(self._name_to_file)

    def get(self, name: str) -> torch.Tensor:
        shard = self._name_to_file[name]
        with self._safe_open(shard, framework="pt") as f:
            return f.get_tensor(name)


def slice_to_geometry(w: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Leading-index slice of ``w`` down to ``target_shape``.

    Raises if any target dim exceeds the source dim: surgery only removes
    capacity, it never invents parameters.
    """
    if w.dim() != len(target_shape):
        raise ValueError(f"rank mismatch: source {tuple(w.shape)} vs target {target_shape}")
    for axis, (src, dst) in enumerate(zip(w.shape, target_shape)):
        if dst > src:
            raise ValueError(
                f"target dim {dst} exceeds source dim {src} on axis {axis}; "
                "surgery cannot fabricate parameters"
            )
    idx = tuple(slice(0, d) for d in target_shape)
    return w[idx].contiguous()


def _find_expert_tensors(
    names: Iterable[str], layer: int
) -> Tuple[str, Dict]:
    """Detect the checkpoint's expert layout for one decoder layer.

    Returns ("fused", {proj: name}) or ("per_expert", {expert: {proj: name}}).
    """
    fused: Dict[str, str] = {}
    per_expert: Dict[int, Dict[str, str]] = {}
    for name in names:
        m = _EXPERT_FUSED_RE.search(name)
        if m and int(m.group("layer")) == layer:
            fused[m.group("proj")] = name
            continue
        m = _EXPERT_PER_TENSOR_RE.search(name)
        if m and int(m.group("layer")) == layer:
            per_expert.setdefault(int(m.group("expert")), {})[m.group("proj")] = name
    if {"gate_up_proj", "down_proj"} <= set(fused):
        return "fused", fused
    if per_expert:
        complete = {
            e: projs
            for e, projs in per_expert.items()
            if {"gate_proj", "up_proj", "down_proj"} <= set(projs)
        }
        if complete:
            return "per_expert", complete
    raise ValueError(
        f"no complete expert FFN tensors found for layer {layer}; "
        "expected fused gate_up_proj/down_proj or per-expert gate/up/down_proj"
    )


def extract_expert_ffn(
    reader: CheckpointReader, layer: int, expert: int
) -> Dict[str, torch.Tensor]:
    """Pull one source expert's FFN as {gate_proj, up_proj, down_proj} weights.

    Shapes returned follow nn.Linear convention: gate/up [ffn, hidden],
    down [hidden, ffn], regardless of the source layout.
    """
    layout, found = _find_expert_tensors(reader.names(), layer)
    if layout == "per_expert":
        if expert not in found:
            raise KeyError(f"expert {expert} not present in layer {layer}")
        return {proj: reader.get(name) for proj, name in found[expert].items()}

    gate_up = reader.get(found["gate_up_proj"])   # [n_exp, hidden, 2*ffn]
    down = reader.get(found["down_proj"])         # [n_exp, ffn, hidden]
    n_exp = gate_up.shape[0]
    if expert >= n_exp:
        raise KeyError(f"expert {expert} out of range (checkpoint has {n_exp})")
    ffn = gate_up.shape[-1] // 2
    gu = gate_up[expert]                          # [hidden, 2*ffn]
    return {
        # transpose to nn.Linear's [out_features, in_features] convention
        "gate_proj": gu[:, :ffn].T.contiguous(),  # [ffn, hidden]
        "up_proj": gu[:, ffn:].T.contiguous(),    # [ffn, hidden]
        "down_proj": down[expert].T.contiguous(), # [hidden, ffn]
    }


def assign_minion_slot(source_expert: int, plan: SurgeryPlan) -> Tuple[str, int]:
    """Round-robin cluster assignment: even -> alpha, odd -> beta."""
    if source_expert >= plan.total_minions:
        raise ValueError(
            f"source expert {source_expert} has no minion slot "
            f"(block holds {plan.total_minions})"
        )
    cluster = "minions_alpha" if source_expert % 2 == 0 else "minions_beta"
    return cluster, source_expert // 2


def build_minion_fragment(
    reader: CheckpointReader,
    plan: SurgeryPlan,
    source_experts: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Produce a state-dict fragment keyed for FeudalPyramidBlock minions.

    Keys look like ``minions_alpha.experts.{m}.gate_proj.weight``; values are
    float32 tensors already sliced to the plan's geometry. Load with
    ``block.load_state_dict(fragment, strict=False)``.
    """
    if source_experts is None:
        source_experts = list(range(plan.total_minions))

    target_shapes = {
        "gate_proj": (plan.ffn_dim, plan.latent_dim),   # [F, L]
        "up_proj": (plan.ffn_dim, plan.latent_dim),     # [F, L]
        "down_proj": (plan.latent_dim, plan.ffn_dim),   # [L, F]
    }
    fragment: Dict[str, torch.Tensor] = {}
    for e in source_experts:
        cluster, slot = assign_minion_slot(e, plan)
        weights = extract_expert_ffn(reader, plan.source_layer, e)
        for proj, w in weights.items():
            sliced = slice_to_geometry(w.float(), target_shapes[proj])
            fragment[f"{cluster}.experts.{slot}.{proj}.weight"] = sliced
    return fragment


def run_surgery(
    checkpoint_path: str | Path, plan: SurgeryPlan, out_path: Optional[str | Path] = None
) -> Dict[str, torch.Tensor]:
    """End-to-end: open checkpoint, build the minion fragment, optionally save it."""
    reader = CheckpointReader(checkpoint_path)
    fragment = build_minion_fragment(reader, plan)
    if out_path is not None:
        torch.save(fragment, out_path)
    return fragment


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", help="safetensors file or sharded checkpoint dir")
    p.add_argument("--layer", type=int, default=0, help="donor decoder layer index")
    p.add_argument("--latent-dim", type=int, default=1280)
    p.add_argument("--ffn-dim", type=int, default=2048)
    p.add_argument("--minions-per-overlord", type=int, default=64)
    p.add_argument("--out", default="surgery_fragment.pt", help="output .pt path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plan = SurgeryPlan(
        latent_dim=args.latent_dim,
        ffn_dim=args.ffn_dim,
        minions_per_overlord=args.minions_per_overlord,
        source_layer=args.layer,
    )
    fragment = run_surgery(args.checkpoint, plan, args.out)
    total = sum(t.numel() for t in fragment.values())
    print(f"wrote {len(fragment)} tensors ({total:,} params) -> {args.out}")


if __name__ == "__main__":
    main()
