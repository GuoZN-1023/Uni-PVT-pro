# models/fusion_model.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .experts import ExpertNetwork
from .gate import GateNetwork


# models/fusion_model.py

class _MoEHead(nn.Module):
    """A 5-expert MoE head (Region 1..5): experts + gate.

    - no/region label is 1-based (1..5)
    - expert index is 0-based (0..4), expert k <-> region (k+1)
    """

    DEFAULT_EXPERT_ORDER = ("gas", "liquid", "transition", "supercritical", "critical")

    def __init__(self, *, input_dim: int, output_dim: int, cfg: dict):
        super().__init__()
        experts_cfg = cfg.get("experts", {}) or {}
        gate_cfg = cfg.get("gate", {}) or {}

        # ✅ 强制 5 个专家
        expert_order = tuple(cfg.get("experts_order", self.DEFAULT_EXPERT_ORDER))
        if len(expert_order) != 5:
            raise ValueError(f"[_MoEHead] Expected 5 experts, got {len(expert_order)}")

        # ✅ 兼容旧配置：第 4 专家旧叫 extra
        alias = {
            "supercritical": ("supercritical", "extra"),
            "transition": ("transition", "phase_change", "phase"),
            "critical": ("critical", "critical_core"),
            "gas": ("gas",),
            "liquid": ("liquid", "liq"),
        }

        resolved = {}
        for name in expert_order:
            found = None
            for k in alias.get(name, (name,)):
                if k in experts_cfg:
                    found = experts_cfg[k]
                    break
            if found is None:
                raise KeyError(f"Missing expert config for '{name}', expected keys: {alias.get(name)}")
            resolved[name] = found

        self.expert_names = list(expert_order)
        self.n_experts = len(self.expert_names)

        self.gate = GateNetwork(
            input_dim=input_dim,
            hidden_layers=gate_cfg.get("hidden_layers", [64, 64]),
            activation=gate_cfg.get("activation", "relu"),
            dropout=gate_cfg.get("dropout", 0.0),
            n_experts=self.n_experts,   # ✅ 5
        )

        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim,
                resolved[name]["hidden_layers"],
                resolved[name]["activation"],
                dropout=resolved[name].get("dropout", 0.0),
                output_dim=output_dim,
            )
            for name in self.expert_names
        ])

    def forward(self, x: torch.Tensor):
        w = self.gate(x)                      # (B,5)
        outs = [e(x) for e in self.experts]   # 5 * (B,T)
        expert_outputs = torch.stack(outs, 1) # (B,5,T)
        fused = torch.sum(w.unsqueeze(-1) * expert_outputs, dim=1)  # (B,T)
        return fused, w, expert_outputs


class FourierStateEmbed(nn.Module):
    """State embedding for (T, P) with optional Fourier features.

    Why: raw (T, P) have very different scales and the function we need is often
    highly non-linear in both variables. A Fourier embedding makes FiLM conditioning
    smoother and easier to learn.
    """

    def __init__(
        self,
        *,
        in_dim: int = 2,
        num_frequencies: int = 8,
        out_dim: int = 128,
        include_input: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)

        feat_dim = (self.in_dim if self.include_input else 0) + self.in_dim * self.num_frequencies * 2
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

        # frequencies: 2^k
        self.register_buffer(
            "freqs",
            torch.tensor([2.0 ** k for k in range(self.num_frequencies)], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, tp: torch.Tensor) -> torch.Tensor:
        """tp: (B,2) where [:,0]=T(K), [:,1]=P(Pa)."""
        if tp.dim() != 2 or tp.size(1) != self.in_dim:
            raise ValueError(f"FourierStateEmbed expects (B,{self.in_dim}), got {tuple(tp.shape)}")

        # Normalize magnitudes a bit to keep sin/cos stable.
        # T is typically ~ [200, 2000], P may be [1e3, 1e8].
        t = tp[:, 0:1] / 1000.0
        p = tp[:, 1:2] / 1.0e6
        x = torch.cat([t, p], dim=1)  # (B,2)

        # (B,2,F)
        angles = x.unsqueeze(-1) * self.freqs.view(1, 1, -1) * (2.0 * math.pi)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        ff = torch.cat([sin, cos], dim=1)  # (B,4,F)
        ff = ff.reshape(tp.size(0), -1)
        if self.include_input:
            ff = torch.cat([x, ff], dim=1)
        return self.proj(ff)


class BoltzmannPooler(nn.Module):
    """Boltzmann pooling over conformer embeddings.

    Inputs:
      z_conf: (B,K,D)
      e_conf: (B,K) relative energies (any consistent unit)
      T:      (B,) temperature in K

    We compute weights w_k ∝ exp(-(e_k - min(e))/ (k_B * T)).
    Here we treat energies as *relative* and assume they're already in kJ/mol.
    If yours are in a different unit, set cfg.mol_encoder.boltzmann_energy_unit_scale.
    """

    def __init__(self, *, energy_unit_scale: float = 1.0):
        super().__init__()
        self.energy_unit_scale = float(energy_unit_scale)
        # k_B in kJ/mol/K == R/1000
        self.register_buffer("kB", torch.tensor(8.314462618e-3, dtype=torch.float32), persistent=False)

    def forward(self, z_conf: torch.Tensor, e_conf: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if z_conf.dim() != 3:
            raise ValueError(f"z_conf must be (B,K,D), got {tuple(z_conf.shape)}")
        if e_conf.dim() != 2:
            raise ValueError(f"e_conf must be (B,K), got {tuple(e_conf.shape)}")
        if T.dim() != 1:
            raise ValueError(f"T must be (B,), got {tuple(T.shape)}")
        if z_conf.size(0) != e_conf.size(0) or z_conf.size(1) != e_conf.size(1):
            raise ValueError(f"Shape mismatch: z_conf={tuple(z_conf.shape)}, e_conf={tuple(e_conf.shape)}")

        # Stabilize: subtract per-molecule min energy.
        e = e_conf * self.energy_unit_scale
        e0 = e.min(dim=1, keepdim=True).values
        de = e - e0

        denom = (self.kB * torch.clamp(T, min=1.0)).unsqueeze(1)  # (B,1)
        logits = -de / denom
        w = torch.softmax(logits, dim=1)  # (B,K)
        z = torch.sum(w.unsqueeze(-1) * z_conf, dim=1)  # (B,D)
        return z


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: z' = gamma(state)*z + beta(state)."""

    def __init__(self, *, state_dim: int, feat_dim: int):
        super().__init__()
        self.film = nn.Linear(state_dim, 2 * feat_dim)
        self.feat_dim = int(feat_dim)

    def forward(self, z: torch.Tensor, state_emb: torch.Tensor) -> torch.Tensor:
        gb = self.film(state_emb)
        gamma, beta = gb[:, : self.feat_dim], gb[:, self.feat_dim :]
        return gamma * z + beta


class FusionModel(nn.Module):
    """
    Your original 4-expert MoE model, extended to support a PINN-style *two-step* prediction path:

      Z prediction:     x -> MoE_Z -> Z
      other properties: x -> MoE_props -> other targets (NOT conditioned on Z as an input)

    This strictly follows the user's requested logic:
      p,T,... -> {Z, other props} with PINN coupling via loss (Z enters constraints, not inputs)

    If PINN is disabled (cfg.pinn.enabled=false), it behaves exactly like the original:
      x -> MoE_all -> all targets
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        model_cfg = cfg.get("model", {}) or {}
        self.input_dim = int(model_cfg.get("input_dim", 0))
        self.output_dim = int(model_cfg.get("output_dim", 1))

        pinn_cfg = (cfg.get("pinn") or {})
        self.pinn_enabled = bool(pinn_cfg.get("enabled", False))

        # ----------------------------
        # Uni-PVT 2.0: cached molecular encoding mode
        # ----------------------------
        mol_cfg = (cfg.get("mol_encoder") or {})
        self.mol_enabled = bool(mol_cfg.get("enabled", False))
        if self.mol_enabled:
            state_cfg = (mol_cfg.get("state_embedding") or {})
            state_dim = int(state_cfg.get("out_dim", 128))
            n_freq = int(state_cfg.get("num_frequencies", 8))

            self.pooler = BoltzmannPooler(
                energy_unit_scale=float(mol_cfg.get("boltzmann_energy_unit_scale", 1.0))
            )
            self.state_embed = FourierStateEmbed(
                in_dim=2,
                num_frequencies=n_freq,
                out_dim=state_dim,
                include_input=bool(state_cfg.get("include_input", True)),
            )

            # Optional: feed phase label (expert_id/no) into FiLM as an embedding.
            self.use_phase_embedding = bool(mol_cfg.get("use_phase_embedding", True))
            phase_dim = int(mol_cfg.get("phase_embed_dim", 16))
            if self.use_phase_embedding:
                n_phases = int(mol_cfg.get("n_phases", 5))  # safe default
                self.phase_embed = nn.Embedding(n_phases, phase_dim)
                film_state_dim = state_dim + phase_dim
            else:
                self.phase_embed = None
                film_state_dim = state_dim

            self.film = FiLM(state_dim=film_state_dim, feat_dim=self.input_dim)

        # ----------------------------
        # Target names/order (optional, but required for cascade mapping)
        # Backward compatible: older configs may store this under `resolved`.
        # ----------------------------
        data_meta = (cfg.get("data_meta") or cfg.get("resolved") or {})
        self.target_cols = list(data_meta.get("target_cols") or [])
        self.z_name = str(pinn_cfg.get("z_name", "Z (-)"))

        # Decide whether to use the two-step cascade
        # We only do cascade if PINN is enabled AND you predict at least one non-Z target.
        self.cascade = False
        if self.pinn_enabled and self.output_dim > 1:
            if not self.target_cols:
                raise ValueError(
                    "[FusionModel] PINN cascade requires cfg.data_meta.target_cols to map Z vs other targets. "
                    "Make sure you train via train.py (it writes config_used.yaml with data_meta)."
                )
            if self.z_name not in self.target_cols:
                raise ValueError(
                    f"[FusionModel] PINN cascade requires '{self.z_name}' included in target_cols/targets."
                )
            self.z_out_idx = int(self.target_cols.index(self.z_name))
            self.other_target_cols = [c for c in self.target_cols if c != self.z_name]
            if len(self.other_target_cols) == 0:
                self.other_target_cols = []
            else:
                self.cascade = True

        # Build heads
        if self.cascade:
            self.head_z = _MoEHead(input_dim=self.input_dim, output_dim=1, cfg=cfg)
            self.head_props = _MoEHead(
                input_dim=self.input_dim, output_dim=len(self.other_target_cols), cfg=cfg
            )
        else:
            self.head_all = _MoEHead(input_dim=self.input_dim, output_dim=self.output_dim, cfg=cfg)

    def _build_x_from_cache(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Convert a dict batch into MoE features x.

        Required keys:
          - z_conf: (B,K,D)
          - e_conf: (B,K)
          - state:  (B,2) [T(K), P(Pa)]
        Optional keys:
          - expert_id: (B,) for phase embedding
        """
        z_conf = batch["z_conf"]
        e_conf = batch["e_conf"]
        state = batch["state"]

        if not torch.is_tensor(z_conf):
            z_conf = torch.as_tensor(z_conf)
        if not torch.is_tensor(e_conf):
            e_conf = torch.as_tensor(e_conf)
        if not torch.is_tensor(state):
            state = torch.as_tensor(state)

        device = next(self.parameters()).device
        z_conf = z_conf.to(device)
        e_conf = e_conf.to(device)
        state = state.to(device)

        T = state[:, 0]
        z = self.pooler(z_conf, e_conf, T)
        s = self.state_embed(state)
        if self.use_phase_embedding and ("expert_id" in batch) and (self.phase_embed is not None):
            eid = batch["expert_id"]
            if not torch.is_tensor(eid):
                eid = torch.as_tensor(eid)
            eid = eid.to(device).view(-1)
            # typical data uses phase id in {1,2,3,4}; map to [0..]
            idx = (eid - 1).clamp(min=0, max=self.phase_embed.num_embeddings - 1)
            pe = self.phase_embed(idx)
            s = torch.cat([s, pe], dim=1)

        x = self.film(z, s)
        return x

    def forward(self, x: Union[torch.Tensor, Dict[str, Any]]) -> dict:
        """
        Always returns a dict for downstream code simplicity.

        Keys:
          fused: (B,T) final fused predictions in the same order as target_cols
          gate_w: dict with 'all' or {'z','props'}
          expert_outputs: dict with 'all' or {'z','props'}
          aux: misc mapping info
        """
        if isinstance(x, dict):
            x = self._build_x_from_cache(x)

        if not self.cascade:
            fused, w, exps = self.head_all(x)
            return {
                "fused": fused,
                "pred": fused,          # ✅ 兼容旧 trainer：常用键名
                "experts": exps,        # ✅ 兼容旧 trainer：需要 (B,4,T)
                "gate_w": {"all": w},
                "expert_outputs": {"all": exps},
                "aux": {
                    "cascade": False,
                    "target_cols": self.target_cols,
                    "z_name": self.z_name,
                },
            }

        # ---- Step 1: predict Z ----
        z_fused, w_z, exps_z = self.head_z(x)  # (B,1), (B,4), (B,4,1)
        z_pred = z_fused[:, 0:1]  # (B,1)

        # ---- Step 2: predict other properties conditioned on Z ----
        p_fused, w_p, exps_p = self.head_props(x)  # (B,To), ...  (NOT conditioned on Z as input)

        # ---- Reassemble fused in original target order ----
        # IMPORTANT: do NOT use `torch.zeros(...)` + in-place assignment to rebuild outputs.
        # That breaks the autograd graph and makes PINN derivatives dZ/dx fail.
        B = x.shape[0]
        T = len(self.target_cols)

        # target_cols without Z, in original order
        other_indices = [i for i, c in enumerate(self.target_cols) if c != self.z_name]

        # -------- differentiable reassemble for fused_all --------
        # Each entry is (B,1), then we cat along dim=1 to preserve graph.
        fused_parts = [None] * T
        fused_parts[self.z_out_idx] = z_pred  # (B,1)
        for j, col_idx in enumerate(other_indices):
            fused_parts[col_idx] = p_fused[:, j:j+1]  # (B,1)
        fused_all = torch.cat(fused_parts, dim=1)  # (B,T)

        # -------- differentiable reassemble for exps_all --------
        n_experts = exps_z.shape[1]  # ✅ 自动 = 5
        exps_parts = [None] * T
        exps_parts[self.z_out_idx] = exps_z  # (B,E,1)
        for j, col_idx in enumerate(other_indices):
            exps_parts[col_idx] = exps_p[:, :, j:j+1]  # (B,E,1)
        exps_all = torch.cat(exps_parts, dim=2)  # (B,E,T)


        return {
            "fused": fused_all,
            "pred": fused_all,        # ✅ 兼容旧 trainer
            "z_pred": z_pred,
            "props_pred": p_fused,

            "experts": exps_all,      # ✅ 兼容旧 trainer：需要 (B,4,T)
            "gate_w": {"z": w_z, "props": w_p},
            "expert_outputs": {"z": exps_z, "props": exps_p, "all": exps_all},
            "aux": {
                "cascade": True,
                "target_cols": self.target_cols,
                "z_name": self.z_name,
                "z_out_idx": self.z_out_idx,
                "other_target_cols": self.other_target_cols,
                "other_indices": other_indices,
            },
        }