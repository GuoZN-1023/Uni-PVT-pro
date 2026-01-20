"""utils/thermo_pinn.py

Physics-informed loss for thermodynamic consistency.

This file now supports TWO input modes:

1) legacy "scaled-features" mode
   - pinn_input = x_scaled with StandardScaler
   - we unscale P/T internally using scaler_mean/scale + feature_cols

2) Uni-PVT 2.0 "state" mode
   - pinn_input = state tensor (B,2) where [:,0]=T(K), [:,1]=P(Pa)
   - no scaler needed; gradients are taken directly w.r.t. T and P

Both modes implement the same residuals (when the corresponding targets exist):
  - d(ln phi)/dp = (Z-1)/p
  - dH/dp = -(R*T^2/p) * dZ/dT
  - dS/dp = -(R/p) * (Z + T*dZ/dT)

To make PINN affordable, we support stochastic subsampling (default 25% of batch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class _Idx:
    z: Optional[int] = None
    lnphi: Optional[int] = None
    h: Optional[int] = None
    s: Optional[int] = None


class ThermoPinnLoss(nn.Module):
    def __init__(
        self,
        cfg: dict,
        *,
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        scaler_mean: Optional[torch.Tensor] = None,
        scaler_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        pinn_cfg = (cfg.get("pinn") or {})

        self.enabled = bool(pinn_cfg.get("enabled", False))
        self.lam = float(pinn_cfg.get("lambda_pinn", 1.0))
        self.subsample_ratio = float(pinn_cfg.get("subsample_ratio", 0.25))

        # choose mode
        input_mode = str(pinn_cfg.get("input_mode", "auto")).lower()
        if input_mode == "auto":
            # if mol_encoder is enabled, default to state mode
            input_mode = "state" if bool((cfg.get("mol_encoder") or {}).get("enabled", False)) else "scaled"
        if input_mode not in {"scaled", "state"}:
            raise ValueError(f"[PINN] Unknown input_mode={input_mode}. Use 'scaled' or 'state'.")
        self.input_mode = input_mode

        # required names
        self.z_name = str(pinn_cfg.get("z_name", "Z (-)"))
        self.lnphi_name = str(pinn_cfg.get("lnphi_name", "lnphi (-)"))
        self.h_name = str(pinn_cfg.get("h_name", "H (J/mol)"))
        self.s_name = str(pinn_cfg.get("s_name", "S (J/mol/K)"))

        # feature names (legacy)
        self.p_feature = str(pinn_cfg.get("p_feature", "P (MPa)"))
        self.t_feature = str(pinn_cfg.get("t_feature", "T (K)"))

        # universal gas constant (J/mol/K)
        self.register_buffer("R", torch.tensor(8.314462618, dtype=torch.float32), persistent=False)

        # legacy scaler params
        self.feature_cols = list(feature_cols or [])
        self.target_cols = list(target_cols or [])
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale

        # set indices for targets
        self.idxs = self._resolve_target_indices(self.target_cols)

        # safety knobs
        self.p_min = float(pinn_cfg.get("p_min", 1.0))
        self.p_unit = str(pinn_cfg.get("p_unit", "Pa"))  # for state mode
        self.p_in_state_is_mpa = bool(pinn_cfg.get("p_in_state_is_mpa", False))

    def _resolve_target_indices(self, target_cols: List[str]) -> _Idx:
        def _find(name: str) -> Optional[int]:
            try:
                return int(target_cols.index(name))
            except Exception:
                return None

        return _Idx(
            z=_find(self.z_name),
            lnphi=_find(self.lnphi_name),
            h=_find(self.h_name),
            s=_find(self.s_name),
        )

    def _subsample(self, n: int, device: torch.device) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        r = max(0.0, min(1.0, float(self.subsample_ratio)))
        m = max(1, int(round(n * r)))
        return torch.randperm(n, device=device)[:m]

    def _unscale_p_t(self, x_scaled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy path: recover T(K), P(Pa) from scaled features."""
        if self.scaler_mean is None or self.scaler_scale is None:
            raise ValueError("[PINN] scaled mode requires scaler_mean/scaler_scale.")
        if not self.feature_cols:
            raise ValueError("[PINN] scaled mode requires feature_cols.")
        if self.p_feature not in self.feature_cols or self.t_feature not in self.feature_cols:
            raise ValueError(
                f"[PINN] scaled mode requires p_feature/t_feature in feature_cols. "
                f"Missing: {[c for c in [self.p_feature, self.t_feature] if c not in self.feature_cols]}"
            )

        p_idx = int(self.feature_cols.index(self.p_feature))
        t_idx = int(self.feature_cols.index(self.t_feature))

        mu = self.scaler_mean.to(x_scaled.device)
        sc = self.scaler_scale.to(x_scaled.device)
        x_raw = x_scaled * sc + mu

        T = x_raw[:, t_idx]
        P_mpa = x_raw[:, p_idx]
        P = P_mpa * 1.0e6  # MPa -> Pa
        return T, P

    def _state_to_p_t(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """State mode: interpret state=(T,P)."""
        if state.dim() != 2 or state.size(1) != 2:
            raise ValueError(f"[PINN] state mode expects (B,2) state=(T,P), got {tuple(state.shape)}")
        T = state[:, 0]
        P = state[:, 1]
        if self.p_in_state_is_mpa:
            P = P * 1.0e6
        return T, P

    def _grad(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]

    def forward(
        self,
        pinn_input: torch.Tensor,
        preds: torch.Tensor,
        *,
        target_cols: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute PINN loss.

        Args:
          pinn_input:
            - scaled mode: x_scaled (B,F)
            - state mode:  state (B,2) with requires_grad=True
          preds: (B,T) predictions aligned with target_cols
        """
        if not self.enabled:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)

        if target_cols is not None:
            self.target_cols = list(target_cols)
            self.idxs = self._resolve_target_indices(self.target_cols)

        B = int(preds.size(0))
        idx = self._subsample(B, preds.device)
        if idx.numel() == 0:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)

        preds_s = preds[idx]
        x_s = pinn_input[idx]

        # Make sure gradients can flow to state variables.
        if not x_s.requires_grad:
            x_s = x_s.detach().requires_grad_(True)

        if self.input_mode == "scaled":
            T, P = self._unscale_p_t(x_s)
        else:
            T, P = self._state_to_p_t(x_s)

        P = torch.clamp(P, min=self.p_min)

        loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
        terms = 0

        # --- residual 1: d(lnphi)/dp = (Z-1)/p ---
        if self.idxs.z is not None and self.idxs.lnphi is not None:
            Z = preds_s[:, self.idxs.z]
            lnphi = preds_s[:, self.idxs.lnphi]

            # gradient w.r.t. P (Pa)
            dlnphi_dP = self._grad(lnphi, P)
            rhs = (Z - 1.0) / P
            loss = loss + torch.mean((dlnphi_dP - rhs) ** 2)
            terms += 1

        # need dZ/dT for H and S constraints
        dZ_dT = None
        if self.idxs.z is not None and (self.idxs.h is not None or self.idxs.s is not None):
            Z = preds_s[:, self.idxs.z]
            dZ_dT = self._grad(Z, T)

        # --- residual 2: dH/dp = -(R*T^2/p) * dZ/dT ---
        if self.idxs.h is not None and dZ_dT is not None:
            H = preds_s[:, self.idxs.h]
            dH_dP = self._grad(H, P)
            rhs = -(self.R * (T ** 2) / P) * dZ_dT
            loss = loss + torch.mean((dH_dP - rhs) ** 2)
            terms += 1

        # --- residual 3: dS/dp = -(R/p) * (Z + T*dZ/dT) ---
        if self.idxs.s is not None and dZ_dT is not None and self.idxs.z is not None:
            S = preds_s[:, self.idxs.s]
            Z = preds_s[:, self.idxs.z]
            dS_dP = self._grad(S, P)
            rhs = -(self.R / P) * (Z + T * dZ_dT)
            loss = loss + torch.mean((dS_dP - rhs) ** 2)
            terms += 1

        if terms == 0:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)
        return self.lam * loss
