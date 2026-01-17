# utils/trainer.py
from __future__ import annotations

import os
import math
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .thermo_pinn import ThermoPinnLoss


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str] | None) -> dict:
    """
    Metrics per target + mean. y_true/y_pred shape: (N, T).
    """
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    T = y_true.shape[1]

    if target_cols is None or len(target_cols) != T:
        target_cols = [f"t{i}" for i in range(T)]

    per = {}
    maes, rmses, r2s = [], [], []
    for j, name in enumerate(target_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(math.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))
        per[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    mean = {
        "MAE": float(np.mean(maes)) if maes else float("nan"),
        "RMSE": float(np.mean(rmses)) if rmses else float("nan"),
        "R2": float(np.mean(r2s)) if r2s else float("nan"),
    }
    return {"per_target": per, "mean": mean}


def _unwrap_dataset(loader) -> Any:
    """
    Get the underlying dataset from a DataLoader (handles DistributedSampler wrapped datasets).
    """
    return getattr(loader, "dataset", None)


def _gate_and_expert_params(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """
    Returns:
      gate_params: parameters belonging to gate / gating networks
      expert_params: parameters belonging to experts

    IMPORTANT:
      FusionModel's gates/experts are inside heads (head_all/head_z/head_props),
      so model.gate/model.experts may not exist. We keep legacy logic and add a robust fallback.
    """
    gate_params: list[torch.nn.Parameter] = []
    expert_params: list[torch.nn.Parameter] = []

    # ---- legacy hooks ----
    if hasattr(model, "gate"):
        gate_params += list(model.gate.parameters())
    if hasattr(model, "gates"):
        gate_params += list(model.gates.parameters())

    if hasattr(model, "experts"):
        try:
            for _, m in model.experts.items():
                expert_params += list(m.parameters())
        except Exception:
            expert_params += list(model.experts.parameters())

    # ---- robust fallback for FusionModel ----
    if len(gate_params) == 0 and len(expert_params) == 0:
        for name, p in model.named_parameters():
            lname = name.lower()
            if (".gate" in lname) or lname.endswith("gate"):
                gate_params.append(p)
            elif (".expert" in lname) or ("expert_" in lname) or (".experts" in lname):
                expert_params.append(p)

    # de-dup
    gate_ids = {id(p) for p in gate_params}
    expert_params = [p for p in expert_params if id(p) not in gate_ids]

    # extreme fallback: if still empty, treat all as experts
    if len(gate_params) == 0 and len(expert_params) == 0:
        expert_params = list(model.parameters())

    return gate_params, expert_params


def _apply_temperature(gate_w: Dict[str, torch.Tensor] | torch.Tensor, temperature: float) -> Dict[str, torch.Tensor] | torch.Tensor:
    """
    Apply softmax temperature to gate weights. Supports dict heads or tensor.
    """
    if temperature is None:
        return gate_w
    t = float(temperature)
    if t <= 0:
        return gate_w

    def _temp_softmax(w: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        w = torch.clamp(w, min=eps)
        logits = torch.log(w)
        logits = logits / t
        return torch.softmax(logits, dim=-1)

    if isinstance(gate_w, dict):
        return {k: _temp_softmax(v) for k, v in gate_w.items()}
    return _temp_softmax(gate_w)


def _experts_from_out(out: Dict[str, Any]) -> torch.Tensor:
    """
    Normalize model outputs to the trainer's expected experts tensor: (N, E, T).

    Supports:
      - out["experts"] = (N,E,T)  (legacy)
      - out["expert_outputs"]["all"] = (N,E,T)  (FusionModel)
      - out["expert_outputs"] = (N,E,T)  (some variants)
    """
    if "experts" in out and out["experts"] is not None:
        return out["experts"]

    eo = out.get("expert_outputs", None)
    if eo is None:
        raise KeyError("[trainer] Model output missing experts. Expected 'experts' or 'expert_outputs'.")

    if isinstance(eo, dict):
        if "all" in eo:
            return eo["all"]
        return next(iter(eo.values()))
    return eo  # assume tensor (N,E,T)


def _gate_from_out(out: Dict[str, Any]) -> torch.Tensor | Dict[str, torch.Tensor]:
    """
    Get gate weights from out, dict or tensor.
    """
    if "gate_w" not in out or out["gate_w"] is None:
        raise KeyError("[trainer] Model output missing 'gate_w'.")
    return out["gate_w"]


def _fused_from_experts(out: Dict[str, Any], *, prefer_fused: bool = True) -> torch.Tensor:
    """
    Compute fused predictions (N,T).

    If prefer_fused=True and out["fused"] exists, returns it.
    Otherwise computes fused = sum_e gate_w * experts.

    Note:
      In Stage2 we may modify gate_w by temperature AFTER forward, so we must set
      prefer_fused=False to recompute fused consistently with the modified gate.
    """
    if prefer_fused and ("fused" in out) and (out["fused"] is not None):
        return out["fused"]

    experts = _experts_from_out(out)  # (N,E,T)
    gate_w = _gate_from_out(out)      # (N,E) or dict

    if isinstance(gate_w, dict):
        gate_w = gate_w.get("all", next(iter(gate_w.values())))
    return torch.einsum("net,ne->nt", experts, gate_w)


def _ddp_info() -> tuple[bool, int, int, bool]:
    ddp_enabled = torch.distributed.is_available() and torch.distributed.is_initialized()
    if ddp_enabled:
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
        is_main = (rank == 0)
    else:
        rank, world_size, is_main = 0, 1, True
    return ddp_enabled, rank, world_size, is_main


def _ddp_reduce_sum(x: float) -> float:
    """DDP-safe scalar sum."""
    ddp_enabled = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not ddp_enabled:
        return float(x)
    t = torch.tensor([float(x)], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return float(t.item())


def _get_cfg(cfg: dict, key_path: str, default=None):
    cur = cfg
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _pinn_active_for_stage(cfg: dict, stage_name: str) -> bool:
    """
    Fully yaml-driven PINN gating.
    stage_name: "pretrain" | "gate_train" | "finetune"
    """
    pinn_cfg = cfg.get("pinn", {}) or {}
    if not bool(pinn_cfg.get("enabled", False)):
        return False

    sched = pinn_cfg.get("schedule", None)
    if isinstance(sched, dict):
        key_map = {
            "pretrain": "apply_in_pretrain",
            "gate_train": "apply_in_gate_train",
            "finetune": "apply_in_finetune",
        }
        k = key_map.get(stage_name, None)
        if k is None:
            return True
        return bool(sched.get(k, True))

    # default behavior (old): Stage2 + Finetune only
    return stage_name in ("gate_train", "finetune")


def _pinn_should_compute_this_step(cfg: dict, global_step: int) -> bool:
    pinn_cfg = cfg.get("pinn", {}) or {}
    every = int(pinn_cfg.get("compute_every_steps", 1) or 1)
    if every <= 1:
        return True
    return (global_step % every) == 0


def _forward_with_optional_pinn_input(model, x: torch.Tensor, *, enable_pinn: bool):
    """
    If enable_pinn=True, forward with x that requires grad and return (out, x_used).
    Otherwise forward with x and return (out, x).
    """
    if enable_pinn:
        x_used = x.detach().requires_grad_(True)
        out = model(x_used)
        return out, x_used
    out = model(x)
    return out, x


@torch.no_grad()
def _eval_loss_and_metrics(
    model,
    loader,
    criterion,
    device,
    *,
    cfg: dict,
    pinn_loss: ThermoPinnLoss | None,
    stage_name: str,
    target_cols: list[str] | None,
) -> tuple[float, dict]:
    """
    Validation:
      - Always computes criterion(preds,y,...) as base loss
      - Optionally adds PINN term to val_loss if:
          pinn.eval_add_to_val_loss: true AND pinn active for this stage
    """
    model.eval()

    add_pinn = bool(_get_cfg(cfg, "pinn.eval_add_to_val_loss", False))
    pinn_active = (pinn_loss is not None) and add_pinn and _pinn_active_for_stage(cfg, stage_name)

    loss_reduction = str(_get_cfg(cfg, "training.eval_loss_reduction", "batch_mean"))
    total_loss = 0.0
    total_n = 0.0
    n_batches = 0

    ys, ps = [], []

    for x, y, expert_id in loader:
        x = x.to(device)
        y = y.to(device)
        expert_id = expert_id.to(device)

        out = model(x)
        preds = _fused_from_experts(out, prefer_fused=True)

        gate_w = out.get("gate_w", None)
        if isinstance(gate_w, dict):
            gate_w = gate_w.get("all", gate_w)

        loss, *_ = criterion(preds, y, expert_id=expert_id, gate_w=gate_w)

        if pinn_active:
            with torch.enable_grad():
                x_pinn = x.detach().requires_grad_(True)
                out_p = model(x_pinn)
                preds_p = _fused_from_experts(out_p, prefer_fused=True)
                pinn_term, _ = pinn_loss(x_pinn, preds_p)
                loss = loss + pinn_term

        b = float(x.shape[0])
        if loss_reduction == "sample_mean":
            total_loss += float(loss.item()) * b
            total_n += b
        else:
            total_loss += float(loss.item())
            n_batches += 1

        ys.append(y.detach().cpu().numpy())
        ps.append(preds.detach().cpu().numpy())

    if loss_reduction == "sample_mean":
        avg_loss = total_loss / max(total_n, 1.0)
    else:
        avg_loss = total_loss / max(n_batches, 1)

    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0, 1))
    y_pred = np.concatenate(ps, axis=0) if ps else np.zeros((0, 1))
    metrics = _compute_metrics(y_true, y_pred, target_cols)
    return float(avg_loss), metrics


def _atomic_save(path: str, state_dict: dict):
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def train_model(model, dataloaders, criterion, optimizer, cfg, device, logger):
    """
    3-stage training loop:
      Stage1: pretrain experts with hard routing by 'no'
      Stage2: train gate(s) (experts frozen by default), with optional temperature schedule
      Finetune: joint training, optional partial unfreeze

    FIXED:
      - Robust gate/expert param discovery for FusionModel
      - Save best checkpoints so export/viz/shap can find them
      - Align Stage1 checkpoint names with export_results.py expectations
    """
    save_dir = cfg["paths"]["save_dir"]
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    ddp_enabled, rank, world_size, is_main = _ddp_info()

    target_cols = cfg.get("targets", None)
    if isinstance(target_cols, list):
        target_cols = list(target_cols)
    else:
        target_cols = None

    gate_params, expert_params = _gate_and_expert_params(model)

    opt_experts = optimizer.get("experts", None) if isinstance(optimizer, dict) else optimizer
    opt_gates = optimizer.get("gate", None) if isinstance(optimizer, dict) else optimizer
    opt_joint = optimizer.get("joint", None) if isinstance(optimizer, dict) else optimizer

    # --- PINN ---
    ds = _unwrap_dataset(train_loader)
    scaler_mean = getattr(ds, "scaler_mean", None)
    scaler_scale = getattr(ds, "scaler_scale", None)
    feature_cols = getattr(ds, "feature_cols", None)
    target_cols_in_ds = getattr(ds, "target_cols", None)

    pinn_loss = ThermoPinnLoss(
        cfg,
        feature_cols=feature_cols,
        target_cols=target_cols_in_ds,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        device=device,
    )

    training = cfg.get("training", {}) or {}
    clip_grad = float(training.get("gradient_clip_norm", training.get("clip_grad_norm", 0.0)) or 0.0)
    if clip_grad <= 0:
        clip_grad = 1e9

    pretrain_epochs = int(training.get("stage1_epochs", training.get("pretrain_epochs", 0)) or 0)

    stage2_epochs = training.get("stage2_epochs", None)
    if stage2_epochs is None:
        stage2_epochs = (training.get("stage2", {}) or {}).get("epochs", None)
    if stage2_epochs is None:
        stage2_epochs = training.get("gate_epochs", None)
    if stage2_epochs is None:
        stage2_epochs = training.get("epochs", 0)
    stage2_epochs = int(stage2_epochs or 0)

    finetune_block = (training.get("finetune", {}) or {})
    finetune_enabled = bool(finetune_block.get("enabled", False))
    finetune_epochs = int(finetune_block.get("epochs", 0) or 0) if finetune_enabled else 0

    temp_sched = (training.get("gate_temperature_schedule", {}) or {})
    temp_enabled = bool(temp_sched.get("enabled", False))
    temp_start = float(temp_sched.get("start", 1.5))
    temp_end = float(temp_sched.get("end", 0.8))

    global_step = 0

    # Best tracking (val_loss by default)
    best_stage2_val = float("inf")
    best_ft_val = float("inf")

    stage2_best_path = os.path.join(ckpt_dir, "best_stage2.pt")
    ft_best_path = os.path.join(ckpt_dir, "best_finetune.pt")
    best_model_path = os.path.join(ckpt_dir, "best_model.pt")  # for shap/export compatibility

    # ---------------- Stage 1: Pretrain experts ----------------
    if pretrain_epochs > 0 and opt_experts is not None:
        if is_main:
            logger.info(f"=== Stage 1: Pretrain experts ({pretrain_epochs} epochs) ===")

        for p in gate_params:
            p.requires_grad = False
        for p in expert_params:
            p.requires_grad = True

        best_stage1 = {1: float("inf"), 2: float("inf"), 3: float("inf"), 4: float("inf")}
        # IMPORTANT: match export_results.py: best_stage1_expert{eid}.pt
        best_stage1_paths = {eid: os.path.join(ckpt_dir, f"best_stage1_expert{eid}.pt") for eid in (1, 2, 3, 4)}

        @torch.no_grad()
        def _eval_stage1_expert_losses():
            model.eval()
            totals = {eid: 0.0 for eid in (1, 2, 3, 4)}
            counts = {eid: 0 for eid in (1, 2, 3, 4)}
            for x, y, expert_id in val_loader:
                x = x.to(device)
                y = y.to(device)
                expert_id = expert_id.to(device).view(-1)

                out = model(x)
                preds_all = _experts_from_out(out)  # (N,E,T)

                for eid in (1, 2, 3, 4):
                    mask = (expert_id == eid)
                    if mask.any():
                        preds = preds_all[mask, eid - 1, :]
                        yy = y[mask, :]
                        loss, *_ = criterion(preds, yy, expert_id=None, gate_w=None)
                        totals[eid] += float(loss.item()) * int(mask.sum().item())
                        counts[eid] += int(mask.sum().item())

            return {eid: (totals[eid] / counts[eid] if counts[eid] > 0 else float("inf")) for eid in (1, 2, 3, 4)}

        for epoch in range(pretrain_epochs):
            model.train()
            running = 0.0
            n_batches = 0

            for x, y, expert_id in train_loader:
                global_step += 1
                x = x.to(device)
                y = y.to(device)
                expert_id = expert_id.to(device)

                stage_name = "pretrain"
                do_pinn = (
                    pinn_loss.enabled
                    and _pinn_active_for_stage(cfg, stage_name)
                    and _pinn_should_compute_this_step(cfg, global_step)
                )

                out, x_used = _forward_with_optional_pinn_input(model, x, enable_pinn=do_pinn)
                experts_out = _experts_from_out(out)  # (N,E,T)

                eidx = (expert_id.view(-1) - 1).clamp(min=0, max=3)
                preds = experts_out[torch.arange(experts_out.size(0), device=device), eidx, :]

                loss, *_ = criterion(preds, y, expert_id=expert_id, gate_w=None)

                if do_pinn:
                    pinn_term, _ = pinn_loss(x_used, preds)
                    loss = loss + pinn_term

                opt_experts.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert_params, clip_grad)
                opt_experts.step()

                running += float(loss.item())
                n_batches += 1

            val_loss, val_metrics = _eval_loss_and_metrics(
                model, val_loader, criterion, device,
                cfg=cfg, pinn_loss=pinn_loss, stage_name="pretrain", target_cols=target_cols_in_ds
            )

            if is_main:
                logger.info(
                    f"[Stage1][{epoch+1}/{pretrain_epochs}] "
                    f"train_loss={running/max(n_batches,1):.6f} "
                    f"val_loss={val_loss:.6f} val_R2_mean={val_metrics['mean']['R2']:.4f}"
                )

            expert_val_losses = _eval_stage1_expert_losses()
            for _eid, _loss in expert_val_losses.items():
                if _loss < best_stage1[_eid]:
                    best_stage1[_eid] = float(_loss)
                    if is_main:
                        _atomic_save(best_stage1_paths[_eid], model.state_dict())

    # ---------------- Stage 2: Train gate(s) ----------------
    if stage2_epochs > 0 and opt_gates is not None:
        if is_main:
            logger.info(f"=== Stage 2: Train gate(s) ({stage2_epochs} epochs) ===")

        parts = _get_cfg(cfg, "training.stage2_train_parts", ["gate"])
        if isinstance(parts, str):
            parts = [parts]
        parts = {str(x).lower() for x in (parts or ["gate"])}

        train_gate = ("gate" in parts) or ("all" in parts)
        train_experts = ("experts" in parts) or ("all" in parts)

        for p in expert_params:
            p.requires_grad = bool(train_experts)
        for p in gate_params:
            p.requires_grad = bool(train_gate)

        stage_name = "gate_train"
        pinn_active = pinn_loss.enabled and _pinn_active_for_stage(cfg, stage_name)

        for epoch in range(stage2_epochs):
            model.train()
            running_sum = 0.0
            running_n = 0.0

            temperature = None
            if temp_enabled and stage2_epochs > 1:
                frac = epoch / float(stage2_epochs - 1)
                temperature = temp_start + frac * (temp_end - temp_start)

            for x, y, expert_id in train_loader:
                global_step += 1
                x = x.to(device)
                y = y.to(device)
                expert_id = expert_id.to(device)

                do_pinn = pinn_active and _pinn_should_compute_this_step(cfg, global_step)
                out, x_used = _forward_with_optional_pinn_input(model, x, enable_pinn=do_pinn)

                gate_w = _gate_from_out(out)
                if isinstance(gate_w, dict):
                    gate_w_for_entropy = {
                        k: (_apply_temperature(v, temperature) if temperature is not None else v)
                        for k, v in gate_w.items()
                    }
                    out["gate_w"] = gate_w_for_entropy
                else:
                    out["gate_w"] = _apply_temperature(gate_w, temperature) if temperature is not None else gate_w

                preds = _fused_from_experts(out, prefer_fused=False)

                loss, data_loss, nonneg, smooth, entropy = criterion(
                    preds,
                    y,
                    expert_id=expert_id,
                    gate_w=out["gate_w"],
                )

                if do_pinn:
                    pinn_term, _ = pinn_loss(x_used, preds)
                    loss = loss + pinn_term

                opt_gates.zero_grad(set_to_none=True)
                loss.backward()

                params_to_clip = []
                if train_gate:
                    params_to_clip += gate_params
                if train_experts:
                    params_to_clip += expert_params
                torch.nn.utils.clip_grad_norm_(params_to_clip, clip_grad)
                opt_gates.step()

                b = float(x.shape[0])
                running_sum += float(loss.item()) * b
                running_n += b

            if ddp_enabled:
                running_sum = _ddp_reduce_sum(running_sum)
                running_n = _ddp_reduce_sum(running_n)

            train_loss = running_sum / max(running_n, 1.0)

            val_loss, val_metrics = _eval_loss_and_metrics(
                model, val_loader, criterion, device,
                cfg=cfg, pinn_loss=pinn_loss, stage_name="gate_train", target_cols=target_cols_in_ds
            )

            if is_main:
                logger.info(
                    f"[Stage2][{epoch+1}/{stage2_epochs}] "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"val_R2_mean={val_metrics['mean']['R2']:.4f}"
                )

            # SAVE BEST Stage2
            if val_loss < best_stage2_val:
                best_stage2_val = float(val_loss)
                if is_main:
                    _atomic_save(stage2_best_path, model.state_dict())
                    logger.info(f"[Stage2] Saved best_stage2.pt (val_loss={best_stage2_val:.6f})")

    # ---------------- Stage 3: Finetune ----------------
    if finetune_enabled and finetune_epochs > 0 and opt_joint is not None:
        if is_main:
            logger.info(f"=== Stage 3: Finetune ({finetune_epochs} epochs) ===")

        unfreeze = finetune_block.get("unfreeze", None)
        if unfreeze is None:
            for p in expert_params:
                p.requires_grad = True
        else:
            # Your FusionModel doesn't expose model.experts dict in a stable way,
            # so do not hard-route here. If you truly need partial unfreeze,
            # do it inside the model by naming parameters and filtering.
            for p in expert_params:
                p.requires_grad = True

        for p in gate_params:
            p.requires_grad = True

        stage_name = "finetune"
        pinn_active = pinn_loss.enabled and _pinn_active_for_stage(cfg, stage_name)

        for epoch in range(finetune_epochs):
            model.train()
            running_sum = 0.0
            running_n = 0.0

            for x, y, expert_id in train_loader:
                global_step += 1
                x = x.to(device)
                y = y.to(device)
                expert_id = expert_id.to(device)

                do_pinn = pinn_active and _pinn_should_compute_this_step(cfg, global_step)
                out, x_used = _forward_with_optional_pinn_input(model, x, enable_pinn=do_pinn)

                preds = _fused_from_experts(out, prefer_fused=True)
                gate_w = out.get("gate_w", None)

                loss, *_ = criterion(preds, y, expert_id=expert_id, gate_w=gate_w)

                if do_pinn:
                    pinn_term, _ = pinn_loss(x_used, preds)
                    loss = loss + pinn_term

                opt_joint.zero_grad(set_to_none=True)
                loss.backward()

                params_to_clip = [p for p in gate_params + expert_params if getattr(p, "requires_grad", False)]
                torch.nn.utils.clip_grad_norm_(params_to_clip, clip_grad)
                opt_joint.step()

                b = float(x.shape[0])
                running_sum += float(loss.item()) * b
                running_n += b

            if ddp_enabled:
                running_sum = _ddp_reduce_sum(running_sum)
                running_n = _ddp_reduce_sum(running_n)

            train_loss = running_sum / max(running_n, 1.0)

            val_loss, val_metrics = _eval_loss_and_metrics(
                model, val_loader, criterion, device,
                cfg=cfg, pinn_loss=pinn_loss, stage_name="finetune", target_cols=target_cols_in_ds
            )

            if is_main:
                logger.info(
                    f"[Finetune][{epoch+1}/{finetune_epochs}] "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"val_R2_mean={val_metrics['mean']['R2']:.4f}"
                )

            # SAVE BEST Finetune + alias best_model.pt
            if val_loss < best_ft_val:
                best_ft_val = float(val_loss)
                if is_main:
                    _atomic_save(ft_best_path, model.state_dict())
                    _atomic_save(best_model_path, model.state_dict())
                    logger.info(f"[Finetune] Saved best_finetune.pt + best_model.pt (val_loss={best_ft_val:.6f})")

    # If finetune disabled but Stage2 exists, provide best_model.pt alias for downstream scripts
    if is_main and (not os.path.exists(best_model_path)) and os.path.exists(stage2_best_path):
        _atomic_save(best_model_path, torch.load(stage2_best_path, map_location=device))
        logger.info("[trainer] Finetune disabled: aliased best_model.pt <- best_stage2.pt")

    if is_main:
        logger.info("=== Training finished ===")