# utils/physics_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """
    Loss = region-weighted(data loss) + lambda_nonneg*nonneg + lambda_smooth*smooth + lambda_entropy*entropy(optional)

    支持多目标回归：
      pred: (B,T)  target: (B,T)
      data loss 会先对 target 维度做 mean，得到每个样本一个标量 loss，再做区域加权
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        tr = cfg.get("training", {}) or {}
        ls = cfg.get("loss", {}) or {}

        # -----------------------------
        # Loss type / core hyperparams
        # -----------------------------
        # Backward compatible:
        # - older configs put these under training.*
        # - newer configs put these under loss.*
        self.loss_type = str(tr.get("loss_type", ls.get("supervised", "mse"))).lower()   # "mse" or "huber"
        self.huber_delta = float(tr.get("huber_delta", ls.get("huber_delta", 1.0)))

        self.lambda_nonneg = float(tr.get("lambda_nonneg", ls.get("lambda_nonneg", 0.0)))
        self.lambda_smooth = float(tr.get("lambda_smooth", ls.get("lambda_smooth", 0.0)))
        self.lambda_entropy = float(tr.get("lambda_entropy", ls.get("lambda_entropy", 0.0)))  # encourage sharp/flat gate

        # -----------------------------
        # Region weights (by expert_col / no)
        # -----------------------------
        # Supported formats:
        # 1) {1: 1.0, 2: 1.2, 3: 1.5, 4: 1.1}
        # 2) {enabled: true/false, weights: {...}}
        rw = tr.get("region_weights", None)
        if rw is None:
            rw = ls.get("region_weights", None)

        self.region_weights = None
        if isinstance(rw, dict):
            # New structured format
            if "enabled" in rw and "weights" in rw:
                if not bool(rw.get("enabled", False)):
                    rw = {}
                else:
                    rw = rw.get("weights", {}) or {}

            # Parse numeric keys only (ignore non-numeric keys safely)
            parsed = {}
            for k, v in (rw or {}).items():
                try:
                    parsed[int(k)] = float(v)
                except Exception:
                    continue
            if parsed:
                self.region_weights = parsed

        # -----------------------------
        # Target scale normalization (multi-target stability)
        # -----------------------------
        # If cfg['loss']['target_scales'] is provided (dict: target_name -> scale),
        # we normalize the supervised residual by scale per target: ((pred-target)/scale)^2.
        # This balances gradients while keeping model outputs in physical units
        # (so PINN constraints continue to make sense).
        self.target_scales = None
        self.target_order = None

        # target order (matches dataset/heads output order)
        resolved = cfg.get('resolved', {}) or {}
        data_meta = cfg.get('data_meta', {}) or {}
        self.target_order = resolved.get('target_cols', None) or data_meta.get('target_cols', None)

        # parse scales
        ts = ls.get('target_scales', None)
        if ts is None:
            ts = tr.get('target_scales', None)
        parsed_scales = {}
        if isinstance(ts, dict):
            for k, v in ts.items():
                try:
                    parsed_scales[str(k)] = float(v)
                except Exception:
                    continue
        if parsed_scales:
            self.target_scales = parsed_scales

    def _data_loss_per_sample(self, pred: torch.Tensor, target: torch.Tensor, *, target_indices=None) -> torch.Tensor:
        """
        返回每个样本一个标量 loss，shape (B,)

        pred/target: (B,T) or (B,1) or (B,)
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        # ----- optional per-target scaling for stability -----
        # scale is aligned to the current pred/target columns
        if self.target_scales is not None:
            names = None
            if target_indices is not None and self.target_order is not None:
                try:
                    names = [self.target_order[int(i)] for i in list(target_indices)]
                except Exception:
                    names = None
            if names is None and self.target_order is not None and pred.size(1) == len(self.target_order):
                names = list(self.target_order)

            if names is not None:
                scales = [float(self.target_scales.get(str(n), 1.0)) for n in names]
                scale_t = pred.new_tensor(scales).view(1, -1)
                scale_t = torch.clamp(scale_t, min=1e-12)
            else:
                # fallback: still protect against division by zero
                scale_t = pred.new_ones((1, pred.size(1)))

            pred = pred / scale_t
            target = target / scale_t

        if self.loss_type == "huber":
            # per-element
            el = F.smooth_l1_loss(pred, target, beta=self.huber_delta, reduction="none")  # (B,T)
        else:
            el = (pred - target) ** 2  # (B,T)

        # mean over targets
        return el.mean(dim=1)  # (B,)

    def forward(
        self,
        fused_pred: torch.Tensor,
        target: torch.Tensor,
        expert_outputs: dict | None = None,
        gate_weights=None,
        *,
        expert_id: torch.Tensor | None = None,
        target_indices=None,
        gate_w=None,
        **kwargs,
    ):
        """
        兼容两种调用：
          - criterion(pred, y, expert_outputs=..., gate_weights=..., expert_id=...)
          - criterion(pred, y, expert_id=..., gate_w=...)

        返回 (total_loss, data_loss, nonneg_term, smooth_term, entropy_term)
        其中 nonneg/smooth/entropy 都是“加权后的贡献项”，相加等于 total_loss。
        """
        # trainer uses gate_w; keep backward-compatibility
        if gate_weights is None:
            gate_weights = gate_w

        # ----- data loss per sample -----
        data_loss_per_sample = self._data_loss_per_sample(fused_pred, target, target_indices=target_indices)  # (B,)

        # ----- region weighting -----
        if self.region_weights is not None and expert_id is not None:
            w = torch.ones_like(data_loss_per_sample)
            eid = expert_id.view(-1)
            for rid, ww in self.region_weights.items():
                w = torch.where(eid == int(rid), w.new_tensor(float(ww)), w)
            data_loss_per_sample = data_loss_per_sample * w

        data_loss = data_loss_per_sample.mean()
        total = data_loss

        # ----- nonneg penalty -----
        nonneg = fused_pred.new_tensor(0.0)
        if self.lambda_nonneg > 0:
            nonneg_raw = F.relu(-fused_pred).mean()
            nonneg = self.lambda_nonneg * nonneg_raw
            total = total + nonneg

        # ----- smooth penalty -----
        smooth = fused_pred.new_tensor(0.0)
        if self.lambda_smooth > 0 and fused_pred.size(0) > 1:
            smooth_raw = (fused_pred[1:] - fused_pred[:-1]).pow(2).mean()
            smooth = self.lambda_smooth * smooth_raw
            total = total + smooth

        # ----- entropy regularization on gate weights -----
        entropy = fused_pred.new_tensor(0.0)
        if self.lambda_entropy != 0 and gate_weights is not None:
            if isinstance(gate_weights, dict):
                ent_list = []
                for _, gw in gate_weights.items():
                    if gw is None:
                        continue
                    p = torch.clamp(gw, 1e-9, 1.0)
                    ent_list.append(-(p * torch.log(p)).sum(dim=1).mean())
                entropy_raw = (sum(ent_list) / len(ent_list)) if ent_list else fused_pred.new_tensor(0.0)
            else:
                p = torch.clamp(gate_weights, 1e-9, 1.0)
                entropy_raw = -(p * torch.log(p)).sum(dim=1).mean()

            entropy = self.lambda_entropy * entropy_raw
            total = total + entropy

        return total, data_loss, nonneg, smooth, entropy
