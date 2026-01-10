# models/fusion_model.py
import torch
import torch.nn as nn
from .experts import ExpertNetwork
from .gate import GateNetwork


class FusionModel(nn.Module):
    """
    四专家 Mixture-of-Experts 模型（支持多目标回归）：

      - 4 个 ExpertNetwork：gas / liquid / critical / extra
      - 1 个 GateNetwork：输出 4 个权重 w (softmax)
      - 每个专家输出 shape (B, T)
      - 融合输出 fused = sum_i w_i * expert_i(x) -> shape (B, T)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        input_dim = int(cfg["model"]["input_dim"])
        output_dim = int(cfg["model"].get("output_dim", 1))

        gas_cfg   = cfg["experts"]["gas"]
        liq_cfg   = cfg["experts"]["liquid"]
        crit_cfg  = cfg["experts"]["critical"]
        extra_cfg = cfg["experts"]["extra"]
        gate_cfg  = cfg["gate"]

        n_experts = 4

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

        self.gate = GateNetwork(
            input_dim,
            gate_cfg["hidden_layers"],
            gate_cfg["activation"],
            gate_cfg.get("dropout", 0.0),
            n_experts=n_experts,
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
          fused: (B,T)
          w: (B,4)
          expert_outputs: (B,4,T)
        """
        w = self.gate(x)  # (B,4)

        out_gas   = self.expert_gas(x)    # (B,T)
        out_liq   = self.expert_liq(x)    # (B,T)
        out_crit  = self.expert_crit(x)   # (B,T)
        out_extra = self.expert_extra(x)  # (B,T)

        expert_outputs = torch.stack([out_gas, out_liq, out_crit, out_extra], dim=1)  # (B,4,T)
        fused = torch.sum(w.unsqueeze(-1) * expert_outputs, dim=1)  # (B,T)
        return fused, w, expert_outputs
