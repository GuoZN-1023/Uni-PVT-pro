
# utils/trainer.py
from __future__ import annotations
import os
import math
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .thermo_pinn import ThermoPinnLoss


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str] | None = None) -> dict:
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    T = yt.shape[1]
    names = target_cols if (target_cols and len(target_cols) == T) else [f"t{i}" for i in range(T)]

    rows = []
    for j, name in enumerate(names):
        ytj = yt[:, j]
        ypj = yp[:, j]
        rows.append({
            "target": name,
            "MAE": float(mean_absolute_error(ytj, ypj)),
            "MSE": float(mean_squared_error(ytj, ypj)),
            "R2": float(r2_score(ytj, ypj)),
        })

    mean_row = {
        "target": "__mean__",
        "MAE": float(np.mean([r["MAE"] for r in rows])),
        "MSE": float(np.mean([r["MSE"] for r in rows])),
        "R2": float(np.mean([r["R2"] for r in rows])),
    }
    return {"per_target": rows, "mean": mean_row}


def _unwrap_dataset(ds):
    # handle torch.utils.data.Subset nesting
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def _gate_and_expert_params(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """
    Return (gate_params, expert_params) for both non-cascade and cascade models.
    """
    gate_params = []
    expert_params = []

    if hasattr(model, "head_all"):
        gate_params += list(model.head_all.gate.parameters())
        expert_params += list(model.head_all.expert_gas.parameters())
        expert_params += list(model.head_all.expert_liq.parameters())
        expert_params += list(model.head_all.expert_crit.parameters())
        expert_params += list(model.head_all.expert_extra.parameters())
        return gate_params, expert_params

    # cascade
    if hasattr(model, "head_z"):
        gate_params += list(model.head_z.gate.parameters())
        expert_params += list(model.head_z.expert_gas.parameters())
        expert_params += list(model.head_z.expert_liq.parameters())
        expert_params += list(model.head_z.expert_crit.parameters())
        expert_params += list(model.head_z.expert_extra.parameters())
    if hasattr(model, "head_props"):
        gate_params += list(model.head_props.gate.parameters())
        expert_params += list(model.head_props.expert_gas.parameters())
        expert_params += list(model.head_props.expert_liq.parameters())
        expert_params += list(model.head_props.expert_crit.parameters())
        expert_params += list(model.head_props.expert_extra.parameters())

    return gate_params, expert_params


def _apply_temperature(w: torch.Tensor, tau: float) -> torch.Tensor:
    """
    tau < 1 => sharper; tau > 1 => smoother.
    """
    tau = float(max(tau, 1e-6))
    w_pow = torch.pow(torch.clamp(w, 1e-12, 1.0), 1.0 / tau)
    w_tau = w_pow / torch.clamp(w_pow.sum(dim=1, keepdim=True), 1e-12)
    return w_tau


def _fused_from_experts(w: torch.Tensor, expert_outputs: torch.Tensor) -> torch.Tensor:
    # w: (B,4), expert_outputs: (B,4,T) -> (B,T)
    return torch.sum(w.unsqueeze(-1) * expert_outputs, dim=1)


@torch.no_grad()
def _eval_loss_and_metrics(model, loader, criterion, device, *, target_cols: list[str] | None):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    n_batches = 0
    for x, y, expert_id in loader:
        x = x.to(device)
        y = y.to(device)
        expert_id = expert_id.to(device)

        out = model(x)
        preds = out["fused"]
        gate_w = out["gate_w"].get("all", None)
        if gate_w is None:
            # cascade: we do not re-apply temperature during eval here; pass both heads for entropy if needed
            gate_w = out["gate_w"]

        loss, *_ = criterion(preds, y, expert_id=expert_id, gate_w=gate_w)
        total_loss += float(loss.item())
        n_batches += 1

        ys.append(y.detach().cpu().numpy())
        ps.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0, 1))
    y_pred = np.concatenate(ps, axis=0) if ps else np.zeros((0, 1))
    metrics = _compute_metrics(y_true, y_pred, target_cols)
    return total_loss / max(n_batches, 1), metrics


def train_model(model, dataloaders, criterion, optimizer, cfg, device, logger):
    """
    3-stage training loop (kept consistent with your existing pipeline):
      Stage1: pretrain experts with hard routing by 'no'
      Stage2: train gate(s) (experts frozen), with optional temperature schedule
      Finetune: joint training, optional partial unfreeze

    PINN thermo constraints are injected in Stage2 and Finetune only (default).
    """
    save_dir = cfg["paths"]["save_dir"]
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    base_ds = _unwrap_dataset(train_loader.dataset)
    feature_cols = list(getattr(base_ds, "feature_cols", []))
    target_cols = list(getattr(base_ds, "target_cols", []))

    # scaler params for PINN conversion
    scaler_mean = torch.tensor(getattr(base_ds, "scaler_mean", np.zeros(len(feature_cols))), dtype=torch.float32)
    scaler_scale = torch.tensor(getattr(base_ds, "scaler_scale", np.ones(len(feature_cols))), dtype=torch.float32)

    pinn_loss = ThermoPinnLoss(
        cfg,
        feature_cols=feature_cols,
        target_cols=target_cols,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        device=device,
    )

    training = cfg.get("training", {}) or {}

    # Backward/forward compatible epoch keys
    pretrain_epochs = int(training.get("stage1_epochs", training.get("pretrain_epochs", 0)))

    # Stage2 (gate training) epochs: support multiple naming styles
    stage2_epochs = training.get("stage2_epochs", None)
    if stage2_epochs is None:
        stage2_epochs = (training.get("stage2", {}) or {}).get("epochs", None)
    if stage2_epochs is None:
        stage2_epochs = training.get("gate_epochs", None)
    if stage2_epochs is None:
        stage2_epochs = training.get("epochs", 0)
    stage2_epochs = int(stage2_epochs)

    # Finetune epochs: support nested finetune block
    finetune_epochs = training.get("finetune_epochs", None)
    if finetune_epochs is None:
        finetune_block = (training.get("finetune", {}) or {})
        if finetune_block.get("enabled", False):
            finetune_epochs = finetune_block.get("epochs", 0)
        else:
            finetune_epochs = 0
    finetune_epochs = int(finetune_epochs)
    base_lr = float(training.get("learning_rate", 1e-3))
    clip_grad = float(training.get("clip_grad_norm", 5.0))
    tau_stage2 = float(training.get("temperature", 1.0))

    # params split
    gate_params, expert_params = _gate_and_expert_params(model)

    # Stage-specific optimizers
    opt_experts = torch.optim.Adam(expert_params, lr=base_lr) if expert_params else None
    opt_gates = torch.optim.Adam(gate_params, lr=base_lr) if gate_params else None
    opt_joint = optimizer  # keep your external optimizer for finetune

    # ---------------- stage-specific best checkpoints ----------------
    # Stage1: best checkpoint per expert (hard routing)
    best_stage1 = {eid: float('inf') for eid in (1,2,3,4)}
    best_stage1_paths = {eid: os.path.join(ckpt_dir, f'best_stage1_expert{eid}.pt') for eid in (1,2,3,4)}

    # Stage2: best gate-only checkpoint (experts frozen)
    best_stage2_val = float('inf')
    best_stage2_path = os.path.join(ckpt_dir, 'best_stage2.pt')

    # Finetune: best joint checkpoint
    best_finetune_val = float('inf')
    best_finetune_path = os.path.join(ckpt_dir, 'best_finetune.pt')

    # Backward compatibility: keep a canonical best_model.pt
    best_model_path = os.path.join(ckpt_dir, 'best_model.pt')

    def _save(tag: str):
        p = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save(model.state_dict(), p)
        return p


    @torch.no_grad()
    def _eval_stage1_expert_losses():
        model.eval()
        totals = {eid: 0.0 for eid in (1,2,3,4)}
        counts = {eid: 0 for eid in (1,2,3,4)}
        for x, y, expert_id in val_loader:
            x = x.to(device)
            y = y.to(device)
            expert_id = expert_id.to(device).view(-1)
            out = model(x)
            for eid in (1,2,3,4):
                mask = (expert_id == eid)
                if not torch.any(mask):
                    continue
                if out['aux']['cascade']:
                    z_idx = int(out['aux']['z_out_idx'])
                    other_inds = list(out['aux']['other_indices'])
                    exps_z = out['expert_outputs']['z']
                    pred_z = exps_z[mask, eid-1, :]
                    y_z = y[mask, z_idx:z_idx+1]
                    loss_z, *_ = criterion(pred_z, y_z, expert_id=expert_id[mask], gate_w=None, target_indices=[z_idx])
                    exps_p = out['expert_outputs']['props']
                    pred_p = exps_p[mask, eid-1, :]
                    y_p = y[mask][:, other_inds]
                    loss_p, *_ = criterion(pred_p, y_p, expert_id=expert_id[mask], gate_w=None, target_indices=list(other_inds))
                    loss = loss_z + loss_p
                else:
                    exps = out['expert_outputs']['all']
                    pred = exps[mask, eid-1, :]
                    loss, *_ = criterion(pred, y[mask], expert_id=expert_id[mask], gate_w=None)
                n = int(mask.sum().item())
                totals[eid] += float(loss.item()) * n
                counts[eid] += n
        return {eid: (totals[eid]/counts[eid] if counts[eid] > 0 else float('inf')) for eid in (1,2,3,4)}

    # ---------------- Stage 1: Pretrain experts (hard routing by expert_id) ----------------
    if pretrain_epochs > 0 and opt_experts is not None:
        logger.info(f"=== Stage 1: Pretrain experts ({pretrain_epochs} epochs) ===")
        # freeze gates
        for p in gate_params:
            p.requires_grad = False
        for p in expert_params:
            p.requires_grad = True

        for epoch in range(pretrain_epochs):
            model.train()
            running = 0.0
            n_batches = 0

            for x, y, expert_id in train_loader:
                x = x.to(device)
                y = y.to(device)
                expert_id = expert_id.to(device).view(-1)  # (B,)

                out = model(x)
                # hard routing: pick expert output based on expert_id
                if out["aux"]["cascade"]:
                    # Z head: target is only Z
                    z_idx = int(out["aux"]["z_out_idx"])
                    y_z = y[:, z_idx:z_idx+1]
                    exps_z = out["expert_outputs"]["z"]  # (B,4,1)
                    sel_z = exps_z[torch.arange(x.shape[0], device=device), torch.clamp(expert_id-1,0,3)]
                    # props head
                    other_inds = out["aux"]["other_indices"]
                    y_p = y[:, other_inds]
                    exps_p = out["expert_outputs"]["props"]  # (B,4,To)
                    sel_p = exps_p[torch.arange(x.shape[0], device=device), torch.clamp(expert_id-1,0,3)]

                    loss_z, *_ = criterion(sel_z, y_z, expert_id=expert_id, gate_w=None, target_indices=[z_idx])
                    loss_p, *_ = criterion(sel_p, y_p, expert_id=expert_id, gate_w=None, target_indices=list(other_inds))
                    loss = loss_z + loss_p
                else:
                    exps = out["expert_outputs"]["all"]  # (B,4,T)
                    sel = exps[torch.arange(x.shape[0], device=device), torch.clamp(expert_id-1,0,3)]
                    loss, *_ = criterion(sel, y, expert_id=expert_id, gate_w=None)

                opt_experts.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert_params, clip_grad)
                opt_experts.step()

                running += float(loss.item())
                n_batches += 1

            val_loss, val_metrics = _eval_loss_and_metrics(model, val_loader, criterion, device, target_cols=target_cols)
            logger.info(f"[Stage1][{epoch+1}/{pretrain_epochs}] train_loss={running/max(n_batches,1):.6f} val_loss={val_loss:.6f} "
                        f"val_R2_mean={val_metrics['mean']['R2']:.4f}")
            expert_val_losses = _eval_stage1_expert_losses()
            for _eid, _loss in expert_val_losses.items():
                if _loss < best_stage1[_eid]:
                    best_stage1[_eid] = float(_loss)
                    torch.save(model.state_dict(), best_stage1_paths[_eid])

    # ---------------- Stage 2: Train gate(s) ----------------
    if stage2_epochs > 0 and opt_gates is not None:
        logger.info(f"=== Stage 2: Train gate(s) ({stage2_epochs} epochs) ===")
        # freeze experts, unfreeze gates
        for p in expert_params:
            p.requires_grad = False
        for p in gate_params:
            p.requires_grad = True

        for epoch in range(stage2_epochs):
            model.train()
            running = 0.0
            n_batches = 0

            for x, y, expert_id in train_loader:
                x = x.to(device)
                y = y.to(device)
                expert_id = expert_id.to(device).view(-1)

                if pinn_loss.enabled:
                    x = x.clone().detach().requires_grad_(True)

                out = model(x)

                if out["aux"]["cascade"]:
                    # apply temperature per head and recompute fused
                    w_z = _apply_temperature(out["gate_w"]["z"], tau_stage2)
                    w_p = _apply_temperature(out["gate_w"]["props"], tau_stage2)
                    z_tau = _fused_from_experts(w_z, out["expert_outputs"]["z"])  # (B,1)
                    p_tau = _fused_from_experts(w_p, out["expert_outputs"]["props"])  # (B,To)

                    # assemble
                    preds = out["fused"].clone()
                    z_idx = int(out["aux"]["z_out_idx"])
                    preds[:, z_idx:z_idx+1] = z_tau
                    preds[:, out["aux"]["other_indices"]] = p_tau

                    gate_w_for_loss = {"z": w_z, "props": w_p}
                else:
                    w_tau = _apply_temperature(out["gate_w"]["all"], tau_stage2)
                    preds = _fused_from_experts(w_tau, out["expert_outputs"]["all"])
                    gate_w_for_loss = w_tau

                loss, data_loss, nonneg, smooth, entropy = criterion(preds, y, expert_id=expert_id, gate_w=gate_w_for_loss)

                pinn_term = torch.tensor(0.0, device=device)
                pinn_parts = {}
                if pinn_loss.enabled:
                    pinn_term, pinn_parts = pinn_loss(x, preds)
                    loss = loss + pinn_term

                opt_gates.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate_params, clip_grad)
                opt_gates.step()

                running += float(loss.item())
                n_batches += 1

            val_loss, val_metrics = _eval_loss_and_metrics(model, val_loader, criterion, device, target_cols=target_cols)
            logger.info(f"[Stage2][{epoch+1}/{stage2_epochs}] train_loss={running/max(n_batches,1):.6f} val_loss={val_loss:.6f} "
                        f"val_R2_mean={val_metrics['mean']['R2']:.4f}")

            if val_loss < best_stage2_val:
                best_stage2_val = val_loss
                torch.save(model.state_dict(), best_stage2_path)

    # ---------------- Finetune: Joint training ----------------
    if finetune_epochs > 0:
        logger.info(f"=== Finetune: Joint training ({finetune_epochs} epochs) ===")
        for p in gate_params + expert_params:
            p.requires_grad = True

        for epoch in range(finetune_epochs):
            model.train()
            running = 0.0
            n_batches = 0

            for x, y, expert_id in train_loader:
                x = x.to(device)
                y = y.to(device)
                expert_id = expert_id.to(device).view(-1)

                if pinn_loss.enabled:
                    x = x.clone().detach().requires_grad_(True)

                out = model(x)
                preds = out["fused"]
                gate_w = out["gate_w"].get("all", None)
                if gate_w is None:
                    gate_w = out["gate_w"]

                loss, *_ = criterion(preds, y, expert_id=expert_id, gate_w=gate_w)
                if pinn_loss.enabled:
                    pinn_term, _ = pinn_loss(x, preds)
                    loss = loss + pinn_term

                opt_joint.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate_params + expert_params, clip_grad)
                opt_joint.step()

                running += float(loss.item())
                n_batches += 1

            val_loss, val_metrics = _eval_loss_and_metrics(model, val_loader, criterion, device, target_cols=target_cols)
            logger.info(f"[Finetune][{epoch+1}/{finetune_epochs}] train_loss={running/max(n_batches,1):.6f} val_loss={val_loss:.6f} "
                        f"val_R2_mean={val_metrics['mean']['R2']:.4f}")

            if val_loss < best_finetune_val:
                best_finetune_val = val_loss
                torch.save(model.state_dict(), best_finetune_path)

    # always save last
    last_path = _save('last_model')

    canonical_src = None
    canonical_val = None
    if finetune_epochs > 0 and os.path.exists(best_finetune_path):
        canonical_src = best_finetune_path
        canonical_val = best_finetune_val
    elif stage2_epochs > 0 and os.path.exists(best_stage2_path):
        canonical_src = best_stage2_path
        canonical_val = best_stage2_val

    if canonical_src is not None:
        try:
            import shutil
            shutil.copyfile(canonical_src, best_model_path)
        except Exception:
            pass
        logger.info(f"Best model saved: {best_model_path} (src={os.path.basename(canonical_src)}, val_loss={float(canonical_val):.6f})")
    else:
        logger.info(f"No Stage2/Finetune best tracked; using last model: {last_path}")

    # Export a small training summary
    summary = {
        "best_val_loss": float(best_finetune_val if (finetune_epochs>0 and not math.isinf(best_finetune_val)) else (best_stage2_val if (stage2_epochs>0 and not math.isinf(best_stage2_val)) else float("inf"))),
        "best_model_path": best_model_path if os.path.exists(best_model_path) else None,
        "target_cols": target_cols,
        "feature_cols": feature_cols,
        "pinn_enabled": bool(pinn_loss.enabled),
        "pinn_cfg": cfg.get("pinn", {}),
    }
    with open(os.path.join(save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
