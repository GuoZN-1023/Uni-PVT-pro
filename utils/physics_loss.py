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

        tr = cfg.get("training", {})
        self.loss_type = str(tr.get("loss_type", "mse")).lower()   # "mse" or "huber"
        self.huber_delta = float(tr.get("huber_delta", 1.0))

        self.lambda_nonneg = float(tr.get("lambda_nonneg", 0.0))
        self.lambda_smooth = float(tr.get("lambda_smooth", 0.0))
        self.lambda_entropy = float(tr.get("lambda_entropy", 0.0))  # encourage sharp gate (optional)

        # region weights: dict like {1:1.0, 2:1.3, 3:1.6, 4:1.2}
        self.region_weights = tr.get("region_weights", None)
        if isinstance(self.region_weights, dict):
            self.region_weights = {int(k): float(v) for k, v in self.region_weights.items()}
        else:
            self.region_weights = None

    def _data_loss_per_sample(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        返回每个样本一个标量 loss，shape (B,)

        pred/target: (B,T) or (B,1) or (B,)
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        if self.loss_type == "huber":
            # per-element
            el = F.smooth_l1_loss(pred, target, beta=self.huber_delta, reduction="none")  # (B,T)
        else:
            el = (pred - target) ** 2  # (B,T)

        return el.mean(dim=1)  # (B,)

    def _region_weight_vec(self, expert_id: torch.Tensor, device) -> torch.Tensor | None:
        """expert_id shape (B,), values 1..4"""
        if self.region_weights is None or expert_id is None:
            return None
        eid = expert_id.view(-1).long().to(device)
        w = torch.ones_like(eid, dtype=torch.float32, device=device)
        for k, v in self.region_weights.items():
            w = torch.where(eid == int(k), torch.tensor(float(v), device=device), w)
        return w

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                gate_w: torch.Tensor | None = None,
                expert_id: torch.Tensor | None = None):
        device = pred.device

        # -------- data loss --------
        loss_vec = self._data_loss_per_sample(pred, target)  # (B,)
        region_w = self._region_weight_vec(expert_id, device)
        if region_w is not None:
            data_loss = torch.mean(loss_vec * region_w)
        else:
            data_loss = torch.mean(loss_vec)

        # -------- nonneg penalty (encourage pred >= 0) --------
        nonneg = torch.tensor(0.0, device=device)
        if self.lambda_nonneg != 0.0:
            nonneg = torch.mean(F.relu(-pred))

        # -------- smooth penalty (batch-adjacent smoothness) --------
        smooth = torch.tensor(0.0, device=device)
        if self.lambda_smooth != 0.0:
            if pred.dim() == 1:
                dp = pred[1:] - pred[:-1]
                smooth = torch.mean(dp ** 2)
            else:
                dp = pred[1:, :] - pred[:-1, :]
                smooth = torch.mean(dp ** 2)

        # -------- entropy regularization on gate weights --------
        entropy = torch.tensor(0.0, device=device)
        if self.lambda_entropy != 0.0 and gate_w is not None:
            w = torch.clamp(gate_w, 1e-12, 1.0)
            ent = -torch.sum(w * torch.log(w), dim=1)  # (B,)
            entropy = torch.mean(ent)

        total = data_loss + self.lambda_nonneg * nonneg + self.lambda_smooth * smooth + self.lambda_entropy * entropy
        return total, data_loss.detach(), nonneg.detach(), smooth.detach(), entropy.detach()
