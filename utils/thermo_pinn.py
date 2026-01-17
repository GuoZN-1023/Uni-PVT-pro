# utils/thermo_pinn.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(x, device: str | torch.device) -> torch.Tensor:
    """
    Robustly convert numpy/torch/scalar to torch.Tensor on device.
    Keeps gradients OFF for scaler stats (they are constants).
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    # numpy.ndarray or python list/float/int
    return torch.as_tensor(x, device=device)


class ThermoPinnLoss(nn.Module):
    """Thermodynamics-informed PINN residuals (soft constraints).

    Enforced relations (at constant T or p):
      (∂ ln φ / ∂ p)_T = (Z - 1) / p
      (∂ H    / ∂ p)_T = -(R T^2 / p) * (∂ Z / ∂ T)_p
      (∂ S    / ∂ p)_T = -(R / p) * ( Z + T (∂ Z / ∂ T)_p )

    Supports two input modes:
      - absolute: features include p_feature and t_feature (e.g. P, T)
      - reduced : features include pr_feature, tr_feature, pc_feature, tc_feature (p_r, T_r, p_c, T_c)

    If pinn.input_mode is missing or set to "auto", mode is inferred from feature_cols.
    For robustness, if a requested mode is incompatible with feature_cols but the other mode is compatible,
    this class will automatically fall back to the compatible mode (no hard failure).
    """

    def __init__(
        self,
        cfg: dict,
        *,
        feature_cols: list[str],
        target_cols: list[str],
        scaler_mean,   # ✅ allow numpy/torch
        scaler_scale,  # ✅ allow numpy/torch
        device: str | torch.device,
    ):
        super().__init__()
        self.cfg = cfg
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)

        pinn = (cfg.get("pinn") or {})
        self.enabled = bool(pinn.get("enabled", True))

        # target names
        self.z_name = str(pinn.get("z_name", "Z (-)"))
        self.phi_name = str(pinn.get("phi_name", "phi (-)"))
        self.h_name = str(pinn.get("h_name", "H (J/mol)"))
        self.s_name = str(pinn.get("s_name", "S (J/mol/K)"))

        # hyperparams
        self.R = float(pinn.get("R", 8.31446261815324))
        self.lambda_phi = float(pinn.get("lambda_phi", 0.0))
        self.lambda_H = float(pinn.get("lambda_H", 0.0))
        self.lambda_S = float(pinn.get("lambda_S", 0.0))

        # scaler params for converting grad wrt x_scaled into grad wrt x_raw
        # ✅ FIX: accept numpy.ndarray as input and convert to torch tensor BEFORE detach
        self.scaler_mean = _as_tensor(scaler_mean, device).detach().float()
        self.scaler_scale = torch.clamp(_as_tensor(scaler_scale, device).detach().float(), min=1e-12)

        # ---- absolute mode config ----
        self.p_feature = str(pinn.get("p_feature", "P (MPa)"))
        self.t_feature = str(pinn.get("t_feature", "T (K)"))
        self.p_to_Pa = float(pinn.get("p_to_Pa", 1.0))  # only used in absolute mode

        # ---- reduced mode config ----
        self.pr_feature = str(pinn.get("pr_feature", "p_r (-)"))
        self.tr_feature = str(pinn.get("tr_feature", "T_r (-)"))
        self.pc_feature = str(pinn.get("pc_feature", "p_c (Pa)"))
        self.tc_feature = str(pinn.get("tc_feature", "T_c (K)"))
        self.pc_to_Pa = float(pinn.get("pc_to_Pa", 1.0))  # only used in reduced mode

        # decide mode
        requested = str(pinn.get("input_mode", "auto")).strip().lower() or "auto"
        has_abs = (self.p_feature in self.feature_cols) and (self.t_feature in self.feature_cols)
        has_red = all(c in self.feature_cols for c in [self.pr_feature, self.tr_feature, self.pc_feature, self.tc_feature])

        if requested == "auto":
            if has_red and not has_abs:
                mode = "reduced"
            elif has_abs:
                mode = "absolute"
            elif has_red:
                mode = "reduced"
            else:
                mode = "absolute"  # will error below with helpful message
        else:
            mode = requested

        # fallback logic (never crash if the other mode is workable)
        if mode == "absolute" and not has_abs and has_red:
            mode = "reduced"
        if mode == "reduced" and not has_red and has_abs:
            mode = "absolute"

        self.input_mode = mode

        # store indices
        if self.input_mode == "absolute":
            missing = [c for c in [self.p_feature, self.t_feature] if c not in self.feature_cols]
            if missing:
                raise ValueError(
                    "[PINN] absolute-mode requires p_feature and t_feature to be included in feature_cols. "
                    f"Missing: {missing}. feature_cols={self.feature_cols}. "
                    "Fix: set pinn.input_mode: reduced (with pr/tr/pc/tc), or include absolute P/T features."
                )
            self.p_idx = int(self.feature_cols.index(self.p_feature))
            self.t_idx = int(self.feature_cols.index(self.t_feature))
            self.pr_idx = self.tr_idx = self.pc_idx = self.tc_idx = None
        else:
            missing = [c for c in [self.pr_feature, self.tr_feature, self.pc_feature, self.tc_feature] if c not in self.feature_cols]
            if missing:
                raise ValueError(
                    "[PINN] reduced-mode requires pr_feature, tr_feature, pc_feature, tc_feature to be included in feature_cols. "
                    f"Missing: {missing}. feature_cols={self.feature_cols}. "
                    "Fix: set pinn.input_mode: absolute (with p_feature/t_feature), or include reduced features."
                )
            self.pr_idx = int(self.feature_cols.index(self.pr_feature))
            self.tr_idx = int(self.feature_cols.index(self.tr_feature))
            self.pc_idx = int(self.feature_cols.index(self.pc_feature))
            self.tc_idx = int(self.feature_cols.index(self.tc_feature))
            self.p_idx = self.t_idx = None

        # target indices (optional; if a target not present, its constraint is skipped)
        self.z_tidx = self.target_cols.index(self.z_name) if self.z_name in self.target_cols else None
        self.phi_tidx = self.target_cols.index(self.phi_name) if self.phi_name in self.target_cols else None
        self.h_tidx = self.target_cols.index(self.h_name) if self.h_name in self.target_cols else None
        self.s_tidx = self.target_cols.index(self.s_name) if self.s_name in self.target_cols else None

    # ---------------------------
    # Gradient unit conversions
    # ---------------------------
    def _dout_dxin_raw_units(self, dout_dxi_scaled: torch.Tensor, idx: int) -> torch.Tensor:
        """Convert derivative wrt x_scaled[:, idx] into derivative wrt x_raw[:, idx]."""
        return dout_dxi_scaled / self.scaler_scale[idx]

    def _dout_dP_SI_from_feature(self, dout_dP_feature_scaled: torch.Tensor, idx: int, *, feature_to_Pa: float) -> torch.Tensor:
        """Convert derivative wrt a pressure-like feature (scaled) into derivative wrt Pa."""
        dout_dP_raw = self._dout_dxin_raw_units(dout_dP_feature_scaled, idx)
        return dout_dP_raw / float(feature_to_Pa)

    # ---------------------------
    # Forward: compute PINN residual loss
    # ---------------------------
    def forward(self, x_scaled: torch.Tensor, pred: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute PINN residual loss.

        x_scaled: (B,F) must have requires_grad=True
        pred: (B,T) fused predictions aligned with self.target_cols
        """
        if (not self.enabled) or (self.lambda_phi == 0 and self.lambda_H == 0 and self.lambda_S == 0):
            return pred.new_tensor(0.0), {}

        if not x_scaled.requires_grad:
            raise ValueError("[PINN] x_scaled must have requires_grad=True to compute thermodynamic residuals.")

        B = pred.shape[0]
        eps = 1e-12

        # Extract predictions needed
        Z = pred[:, self.z_tidx] if self.z_tidx is not None else None
        phi = pred[:, self.phi_tidx] if self.phi_tidx is not None else None
        H = pred[:, self.h_tidx] if self.h_tidx is not None else None
        S = pred[:, self.s_tidx] if self.s_tidx is not None else None

        # Gradients wrt inputs
        details = {}
        total = pred.new_tensor(0.0)

        # Build p (Pa) and T (K) in physical units from inputs
        if self.input_mode == "absolute":
            # raw units
            P_raw = x_scaled[:, self.p_idx] * self.scaler_scale[self.p_idx] + self.scaler_mean[self.p_idx]
            T_raw = x_scaled[:, self.t_idx] * self.scaler_scale[self.t_idx] + self.scaler_mean[self.t_idx]
            p_Pa = P_raw * float(self.p_to_Pa)
            T_K = T_raw  # assume already in K
            # For absolute mode, derivative wrt p(Pa) from feature is handled by p_to_Pa
            p_feature_to_Pa = float(self.p_to_Pa)
            p_like_idx = self.p_idx
            # For d/dT, feature is already K
            dT_factor = 1.0
            T_like_idx = self.t_idx
        else:
            pr_raw = x_scaled[:, self.pr_idx] * self.scaler_scale[self.pr_idx] + self.scaler_mean[self.pr_idx]
            tr_raw = x_scaled[:, self.tr_idx] * self.scaler_scale[self.tr_idx] + self.scaler_mean[self.tr_idx]
            pc_raw = x_scaled[:, self.pc_idx] * self.scaler_scale[self.pc_idx] + self.scaler_mean[self.pc_idx]
            tc_raw = x_scaled[:, self.tc_idx] * self.scaler_scale[self.tc_idx] + self.scaler_mean[self.tc_idx]

            pc_Pa = pc_raw * float(self.pc_to_Pa)
            p_Pa = pr_raw * pc_Pa
            T_K = tr_raw * tc_raw

            # d/dp(Pa) from d/dpr: dp/dpr = pc_Pa => d/dp = (1/pc_Pa) d/dpr
            p_feature_to_Pa = 1.0  # handled separately
            p_like_idx = self.pr_idx
            # d/dT(K) from d/dTr: dT/dTr = Tc => d/dT = (1/Tc) d/dTr
            dT_factor = None
            T_like_idx = self.tr_idx

        p_safe = torch.clamp(p_Pa, min=1e-6)

        # ---------- Constraint 1: d ln(phi) / d p ----------
        if (phi is not None) and (Z is not None) and (self.lambda_phi != 0):
            lnphi = torch.log(torch.clamp(phi, min=1e-12))
            grad_lnphi = torch.autograd.grad(
                lnphi.sum(), x_scaled, create_graph=True, retain_graph=True
            )[0]  # (B,F)
            dlnphi_dx_scaled = grad_lnphi[:, p_like_idx]

            if self.input_mode == "absolute":
                dlnphi_dp = self._dout_dP_SI_from_feature(dlnphi_dx_scaled, p_like_idx, feature_to_Pa=p_feature_to_Pa)
            else:
                # d/dpr_raw
                dlnphi_dpr = self._dout_dxin_raw_units(dlnphi_dx_scaled, p_like_idx)
                pc_raw = x_scaled[:, self.pc_idx] * self.scaler_scale[self.pc_idx] + self.scaler_mean[self.pc_idx]
                pc_Pa = pc_raw * float(self.pc_to_Pa)
                dlnphi_dp = dlnphi_dpr / torch.clamp(pc_Pa, min=1e-12)

            rhs = (Z - 1.0) / p_safe
            res = dlnphi_dp - rhs
            term = (res ** 2).mean()
            total = total + float(self.lambda_phi) * term
            details["pinn_phi_mse"] = term.detach()
            details["pinn_phi_mae"] = res.abs().mean().detach()

        # ---------- compute dZ/dT (K) at constant p ----------
        dZ_dT = None
        if Z is not None:
            grad_Z = torch.autograd.grad(
                Z.sum(), x_scaled, create_graph=True, retain_graph=True
            )[0]
            dZ_dTfeat_scaled = grad_Z[:, T_like_idx]
            if self.input_mode == "absolute":
                # dZ/dT_raw (K)
                dZ_dT = self._dout_dxin_raw_units(dZ_dTfeat_scaled, T_like_idx) * float(dT_factor)
            else:
                dZ_dTr = self._dout_dxin_raw_units(dZ_dTfeat_scaled, T_like_idx)
                tc_raw = x_scaled[:, self.tc_idx] * self.scaler_scale[self.tc_idx] + self.scaler_mean[self.tc_idx]
                dZ_dT = dZ_dTr / torch.clamp(tc_raw, min=1e-12)

        # ---------- Constraint 2: dH/dp ----------
        if (H is not None) and (Z is not None) and (dZ_dT is not None) and (self.lambda_H != 0):
            grad_H = torch.autograd.grad(
                H.sum(), x_scaled, create_graph=True, retain_graph=True
            )[0]
            dH_dx_scaled = grad_H[:, p_like_idx]

            if self.input_mode == "absolute":
                dH_dp = self._dout_dP_SI_from_feature(dH_dx_scaled, p_like_idx, feature_to_Pa=p_feature_to_Pa)
            else:
                dH_dpr = self._dout_dxin_raw_units(dH_dx_scaled, p_like_idx)
                pc_raw = x_scaled[:, self.pc_idx] * self.scaler_scale[self.pc_idx] + self.scaler_mean[self.pc_idx]
                pc_Pa = pc_raw * float(self.pc_to_Pa)
                dH_dp = dH_dpr / torch.clamp(pc_Pa, min=1e-12)

            rhs = -(self.R * (T_K ** 2) / p_safe) * dZ_dT
            res = dH_dp - rhs
            term = (res ** 2).mean()
            total = total + float(self.lambda_H) * term
            details["pinn_H_mse"] = term.detach()
            details["pinn_H_mae"] = res.abs().mean().detach()

        # ---------- Constraint 3: dS/dp ----------
        if (S is not None) and (Z is not None) and (dZ_dT is not None) and (self.lambda_S != 0):
            grad_S = torch.autograd.grad(
                S.sum(), x_scaled, create_graph=True, retain_graph=True
            )[0]
            dS_dx_scaled = grad_S[:, p_like_idx]

            if self.input_mode == "absolute":
                dS_dp = self._dout_dP_SI_from_feature(dS_dx_scaled, p_like_idx, feature_to_Pa=p_feature_to_Pa)
            else:
                dS_dpr = self._dout_dxin_raw_units(dS_dx_scaled, p_like_idx)
                pc_raw = x_scaled[:, self.pc_idx] * self.scaler_scale[self.pc_idx] + self.scaler_mean[self.pc_idx]
                pc_Pa = pc_raw * float(self.pc_to_Pa)
                dS_dp = dS_dpr / torch.clamp(pc_Pa, min=1e-12)

            rhs = -(self.R / p_safe) * (Z + T_K * dZ_dT)
            res = dS_dp - rhs
            term = (res ** 2).mean()
            total = total + float(self.lambda_S) * term
            details["pinn_S_mse"] = term.detach()
            details["pinn_S_mae"] = res.abs().mean().detach()

        return total, details