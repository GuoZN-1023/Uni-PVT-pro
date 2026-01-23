from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
        if input_mode in {"scaled_features", "scaled-feature", "scaledfeatures"}:
            input_mode = "scaled"
        if input_mode == "auto":
            input_mode = "state" if bool((cfg.get("mol_encoder") or {}).get("enabled", False)) else "scaled"
        if input_mode not in {"scaled", "state"}:
            raise ValueError(f"[PINN] Unknown input_mode={input_mode}. Use 'scaled' or 'state'.")
        self.input_mode = input_mode

        # target names
        self.z_name = str(pinn_cfg.get("z_name", "Z (-)"))
        self.lnphi_name = str(pinn_cfg.get("lnphi_name", "lnphi (-)"))
        self.h_name = str(pinn_cfg.get("h_name", "H (J/mol)"))
        self.s_name = str(pinn_cfg.get("s_name", "S (J/mol/K)"))

        # state feature names for scaled mode (direct)
        self.p_feature = str(pinn_cfg.get("p_feature", "P (MPa)"))
        self.t_feature = str(pinn_cfg.get("t_feature", "T (K)"))

        # reduced-state fallback (scaled mode): T = Tr*Tc; P(Pa) = pr*Pc(Pa)
        self.tr_feature = str(pinn_cfg.get("tr_feature", "T_r (-)"))
        self.tc_feature = str(pinn_cfg.get("tc_feature", "T_c (K)"))
        self.pr_feature = str(pinn_cfg.get("pr_feature", "p_r (-)"))
        self.pc_feature = str(pinn_cfg.get("pc_feature", "p_c (Pa)"))

        # constants
        self.register_buffer("R", torch.tensor(8.314462618, dtype=torch.float32), persistent=False)

        # data descriptors
        self.feature_cols = list(feature_cols or [])
        self.target_cols = list(target_cols or [])
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale

        # indices
        self.idxs = self._resolve_target_indices(self.target_cols)

        # safety knobs
        self.p_min = float(pinn_cfg.get("p_min", 1.0))  # Pa
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

    def _scaled_state_from_x(self, x_scaled: torch.Tensor) -> dict:
        if self.scaler_mean is None or self.scaler_scale is None:
            raise ValueError("[PINN] scaled mode requires scaler_mean/scaler_scale from dataset.")
        if not self.feature_cols:
            raise ValueError("[PINN] scaled mode requires feature_cols from dataset.")

        mu = self.scaler_mean.to(x_scaled.device)
        sc = self.scaler_scale.to(x_scaled.device)
        x_raw = x_scaled * sc + mu

        # try direct (T,P) first
        if self.t_feature in self.feature_cols and self.p_feature in self.feature_cols:
            t_idx = int(self.feature_cols.index(self.t_feature))
            p_idx = int(self.feature_cols.index(self.p_feature))
            T = x_raw[:, t_idx]
            P = x_raw[:, p_idx]
            if "MPa" in self.p_feature or "mpa" in self.p_feature:
                P = P * 1.0e6
            dT_dx = sc[t_idx]          # dT/dx_t
            dP_dx = sc[p_idx] * (1.0e6 if ("MPa" in self.p_feature or "mpa" in self.p_feature) else 1.0)  # dP(Pa)/dx_p
            return {"T": T, "P": P, "t_col_idx": t_idx, "p_col_idx": p_idx, "dT_dx": dT_dx, "dP_dx": dP_dx}

        # fallback reduced-state
        needed = [self.tr_feature, self.tc_feature, self.pr_feature, self.pc_feature]
        missing = [c for c in needed if c not in self.feature_cols]
        if missing:
            raise ValueError(
                f"[PINN] scaled mode needs either direct ({self.t_feature}, {self.p_feature}) or "
                f"reduced-state ({', '.join(needed)}). Missing: {missing}"
            )

        tr_idx = int(self.feature_cols.index(self.tr_feature))
        tc_idx = int(self.feature_cols.index(self.tc_feature))
        pr_idx = int(self.feature_cols.index(self.pr_feature))
        pc_idx = int(self.feature_cols.index(self.pc_feature))

        Tr = x_raw[:, tr_idx]
        Tc = x_raw[:, tc_idx]
        pr = x_raw[:, pr_idx]
        Pc = x_raw[:, pc_idx]  # Pa

        T = Tr * Tc
        P = pr * Pc

        dT_dx = sc[tr_idx] * Tc      # dT/dx_tr
        dP_dx = sc[pr_idx] * Pc      # dP(Pa)/dx_pr

        return {"T": T, "P": P, "t_col_idx": tr_idx, "p_col_idx": pr_idx, "dT_dx": dT_dx, "dP_dx": dP_dx}

    def _state_to_p_t(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # NOTE: preds was produced by model forward using the full pinn_input tensor.
        # Do NOT slice pinn_input first and then differentiate w.r.t. that slice; autograd may
        # report the sliced tensor was not used in the graph. Differentiate w.r.t x_full and
        # then take idx rows from the resulting gradient.
        x_full = pinn_input
        if not x_full.requires_grad:
            x_full.requires_grad_(True)
        x_s = x_full[idx]

        if self.input_mode == "scaled":
            st = self._scaled_state_from_x(x_s)
            T = st["T"]
            P = st["P"]
            t_col = int(st["t_col_idx"])
            p_col = int(st["p_col_idx"])
            dT_dx = st["dT_dx"]
            dP_dx = st["dP_dx"]
            t_chain = 1.0 / torch.clamp(dT_dx, min=1e-12)
            p_chain = 1.0 / torch.clamp(dP_dx, min=1e-12)
        else:
            T, P = self._state_to_p_t(x_s)
            t_col = 0
            p_col = 1
            t_chain = torch.ones_like(T)
            p_chain = torch.full_like(P, 1.0e-6) if self.p_in_state_is_mpa else torch.ones_like(P)

        P = torch.clamp(P, min=self.p_min)

        loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
        terms = 0

        # residual 1: d(lnphi)/dP = (Z-1)/P
        if self.idxs.z is not None and self.idxs.lnphi is not None:
            Z = preds_s[:, self.idxs.z]
            lnphi = preds_s[:, self.idxs.lnphi]
            dlnphi_dx_full = self._grad(lnphi, x_full)
            dlnphi_dx = dlnphi_dx_full[idx]
            dlnphi_dP = dlnphi_dx[:, p_col] * p_chain
            rhs = (Z - 1.0) / P
            loss = loss + torch.mean((dlnphi_dP - rhs) ** 2)
            terms += 1

        # dZ/dT for H,S residuals
        dZ_dT = None
        if self.idxs.z is not None and (self.idxs.h is not None or self.idxs.s is not None):
            Z = preds_s[:, self.idxs.z]
            dZ_dx_full = self._grad(Z, x_full)
            dZ_dx = dZ_dx_full[idx]
            dZ_dT = dZ_dx[:, t_col] * t_chain

        # residual 2: dH/dP = -(R*T^2/P) dZ/dT
        if self.idxs.h is not None and dZ_dT is not None:
            H = preds_s[:, self.idxs.h]
            dH_dx_full = self._grad(H, x_full)
            dH_dx = dH_dx_full[idx]
            dH_dP = dH_dx[:, p_col] * p_chain
            rhs = -(self.R * (T ** 2) / P) * dZ_dT
            loss = loss + torch.mean((dH_dP - rhs) ** 2)
            terms += 1

        # residual 3: dS/dP = -(R/P) (Z + T dZ/dT)
        if self.idxs.s is not None and dZ_dT is not None and self.idxs.z is not None:
            S = preds_s[:, self.idxs.s]
            Z = preds_s[:, self.idxs.z]
            dS_dx_full = self._grad(S, x_full)
            dS_dx = dS_dx_full[idx]
            dS_dP = dS_dx[:, p_col] * p_chain
            rhs = -(self.R / P) * (Z + T * dZ_dT)
            loss = loss + torch.mean((dS_dP - rhs) ** 2)
            terms += 1

        if terms == 0:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)
        return self.lam * loss