
# models/fusion_model.py
from __future__ import annotations
import torch
import torch.nn as nn

from .experts import ExpertNetwork
from .gate import GateNetwork


class _MoEHead(nn.Module):
    """A 4-expert MoE head: experts + gate, producing fused output and per-expert outputs."""
    def __init__(self, *, input_dim: int, output_dim: int, cfg: dict):
        super().__init__()
        experts_cfg = cfg.get("experts", {}) or {}
        gate_cfg = cfg.get("gate", {}) or {}

        gas_cfg = experts_cfg["gas"]
        liq_cfg = experts_cfg["liquid"]
        crit_cfg = experts_cfg["critical"]
        extra_cfg = experts_cfg["extra"]

        self.gate = GateNetwork(
            input_dim=input_dim,
            hidden_layers=gate_cfg.get("hidden_layers", [64, 64]),
            activation=gate_cfg.get("activation", "relu"),
            dropout=gate_cfg.get("dropout", 0.0),
            n_experts=4,
        )

        self.expert_gas = ExpertNetwork(
            input_dim,
            gas_cfg["hidden_layers"],
            gas_cfg["activation"],
            dropout=gas_cfg.get("dropout", 0.0),
            output_dim=output_dim,
        )
        self.expert_liq = ExpertNetwork(
            input_dim,
            liq_cfg["hidden_layers"],
            liq_cfg["activation"],
            dropout=liq_cfg.get("dropout", 0.0),
            output_dim=output_dim,
        )
        self.expert_crit = ExpertNetwork(
            input_dim,
            crit_cfg["hidden_layers"],
            crit_cfg["activation"],
            dropout=crit_cfg.get("dropout", 0.0),
            output_dim=output_dim,
        )
        self.expert_extra = ExpertNetwork(
            input_dim,
            extra_cfg["hidden_layers"],
            extra_cfg["activation"],
            dropout=extra_cfg.get("dropout", 0.0),
            output_dim=output_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          fused: (B,T)
          w: (B,4)
          expert_outputs: (B,4,T)
        """
        w = self.gate(x)  # (B,4)
        out_gas = self.expert_gas(x)
        out_liq = self.expert_liq(x)
        out_crit = self.expert_crit(x)
        out_extra = self.expert_extra(x)
        expert_outputs = torch.stack([out_gas, out_liq, out_crit, out_extra], dim=1)  # (B,4,T)
        fused = torch.sum(w.unsqueeze(-1) * expert_outputs, dim=1)  # (B,T)
        return fused, w, expert_outputs


class FusionModel(nn.Module):
    """
    Your original 4-expert MoE model, extended to support a PINN-style *two-step* prediction path:

      Z prediction:     x -> MoE_Z -> Z
      other properties: x -> MoE_Z -> Z, then concat([x, Z]) -> MoE_props -> other targets

    This strictly follows the user's requested logic:
      p,T,... -> Z -> (PINN-embedded) -> other thermo properties

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

        # Target names/order (optional, but required for cascade mapping)
        # Backward compatible: older configs may store this under `resolved`.
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
                raise ValueError(f"[FusionModel] PINN cascade requires '{self.z_name}' included in target_cols/targets.")
            self.z_out_idx = int(self.target_cols.index(self.z_name))
            self.other_target_cols = [c for c in self.target_cols if c != self.z_name]
            if len(self.other_target_cols) == 0:
                # output_dim>1 but somehow only Z? shouldn't happen
                self.other_target_cols = []
            else:
                self.cascade = True

        # Build heads
        if self.cascade:
            # Z head outputs 1 dim
            self.head_z = _MoEHead(input_dim=self.input_dim, output_dim=1, cfg=cfg)
            # props head takes x concatenated with predicted Z
            self.head_props = _MoEHead(input_dim=self.input_dim + 1, output_dim=len(self.other_target_cols), cfg=cfg)
        else:
            # single head outputs all targets (original behavior)
            self.head_all = _MoEHead(input_dim=self.input_dim, output_dim=self.output_dim, cfg=cfg)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Always returns a dict for downstream code simplicity.

        Keys:
          fused: (B,T) final fused predictions in the same order as target_cols
          gate_w: dict with 'all' or {'z','props'}
          expert_outputs: dict with 'all' or {'z','props'}
          aux: misc mapping info
        """
        if not self.cascade:
            fused, w, exps = self.head_all(x)
            return {
                "fused": fused,
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
        x2 = torch.cat([x, z_pred], dim=1)  # (B, F+1)
        p_fused, w_p, exps_p = self.head_props(x2)  # (B,To), ...

        # ---- Reassemble fused in original target order ----
        B = x.shape[0]
        T = len(self.target_cols)
        fused_all = torch.zeros((B, T), device=x.device, dtype=z_pred.dtype)

        # put Z
        fused_all[:, self.z_out_idx:self.z_out_idx + 1] = z_pred

        # put others in order (target_cols without Z)
        other_indices = [i for i, c in enumerate(self.target_cols) if c != self.z_name]
        fused_all[:, other_indices] = p_fused

        # Reassemble expert outputs to (B,4,T)
        exps_all = torch.zeros((B, 4, T), device=x.device, dtype=z_pred.dtype)
        exps_all[:, :, self.z_out_idx:self.z_out_idx + 1] = exps_z  # (B,4,1)

        exps_all[:, :, other_indices] = exps_p  # (B,4,To)

        return {
            "fused": fused_all,
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
