import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch

from utils.dataset import ZDataset
from models.fusion_model import FusionModel


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def _metrics_per_target(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for j, t in enumerate(target_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        rows.append({
            'target': t,
            'R2': _r2(yt, yp),
            'MAE': float(np.mean(np.abs(yp - yt))),
            'MSE': float(np.mean((yp - yt) ** 2)),
            'N': int(len(yt)),
        })
    return pd.DataFrame(rows)


def _split_indices(n_total: int, seed: int = 42):
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    # emulate torch.utils.data.random_split index generation
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _make_loader(dataset, indices, batch_size: int):
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)


def _expert_pred_from_out(out: dict, eid: int) -> torch.Tensor:
    """Return predictions from a single expert (hard routed) for all samples in batch."""
    idx = eid - 1
    if out["aux"].get("cascade", False):
        z_idx = int(out["aux"]["z_out_idx"])
        other_inds = list(out["aux"]["other_indices"])
        B = out["fused"].shape[0]
        T = out["fused"].shape[1]
        pred = torch.zeros((B, T), device=out["fused"].device, dtype=out["fused"].dtype)
        pred[:, z_idx:z_idx + 1] = out["expert_outputs"]["z"][:, idx, :]
        pred[:, other_inds] = out["expert_outputs"]["props"][:, idx, :]
        return pred
    else:
        return out["expert_outputs"]["all"][:, idx, :]


def _eval_fused(model, loader, device):
    ys, ps, eids = [], [], []
    for x, y, expert_id in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model(x)
            pred = out["fused"]
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
        eids.append(expert_id.detach().cpu().numpy().reshape(-1))
    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(ps, axis=0)
    e_all = np.concatenate(eids, axis=0)
    return y_all, p_all, e_all


def _eval_expert(model, loader, device, eid: int):
    ys, ps, eids = [], [], []
    for x, y, expert_id in loader:
        expert_id = expert_id.view(-1)
        mask = (expert_id == eid)
        if not torch.any(mask):
            continue
        x = x.to(device)
        y = y.to(device)
        expert_id = expert_id.to(device)
        with torch.no_grad():
            out = model(x)
            pred_all = _expert_pred_from_out(out, eid)
            pred = pred_all[mask.to(device)]
        ys.append(y[mask.to(device)].detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
        eids.append(expert_id[mask.to(device)].detach().cpu().numpy().reshape(-1))
    if not ys:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0,))
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0), np.concatenate(eids, axis=0)


def _attach_true_pred(df_base: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str]) -> pd.DataFrame:
    df = df_base.copy()
    for j, t in enumerate(target_cols):
        df[f"y_true::{t}"] = y_true[:, j]
        df[f"y_pred::{t}"] = y_pred[:, j]
    return df


def _set_or_move_first(df: pd.DataFrame, col: str, values=None) -> pd.DataFrame:
    """Set df[col]=values (if provided) and make sure col is the first column."""
    if values is not None:
        df[col] = values
    if col in df.columns:
        cols = [col] + [c for c in df.columns if c != col]
        return df[cols]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='config_used.yaml (or any config with paths)')
    ap.add_argument('--outdir', default=None, help='output folder for the 6 CSVs (default: <save_dir>/exports)')
    ap.add_argument('--device', default=None, help='cpu/cuda')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    save_dir = (cfg.get('paths', {}) or {}).get('save_dir', None)
    if not save_dir:
        raise ValueError('config missing paths.save_dir')

    outdir = args.outdir or os.path.join(save_dir, 'exports')
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    data_path = cfg['paths']['data']
    scaler_path = cfg['paths']['scaler']

    # Build dataset in inference mode (loads scaler)
    ds = ZDataset(csv_path=data_path, scaler_path=scaler_path, cfg=cfg, train=False)

    target_cols = list(ds.target_cols)
    expert_col = ds.expert_col

    # Split indices deterministically
    train_idx, val_idx, test_idx = _split_indices(len(ds), seed=42)

    bs = int(((cfg.get('training', {}) or {}).get('batch_size', 256)))
    test_loader = _make_loader(ds, test_idx, batch_size=bs)

    # Keep outputs tidy: only keep expert_col + feature columns (drop raw targets etc.)
    base_cols = []
    base_cols.append(expert_col)
    for c in list(ds.feature_cols):
        if c not in base_cols:
            base_cols.append(c)
    df_test_base = ds.num_df.iloc[test_idx][base_cols].reset_index(drop=True)

    # Build model
    model = FusionModel(cfg).to(device)

    ckpt_dir = os.path.join(save_dir, 'checkpoints')

    # -------------------- Stage2 (gate-only) --------------------
    stage2_ckpt = os.path.join(ckpt_dir, 'best_stage2.pt')
    if os.path.exists(stage2_ckpt):
        model.load_state_dict(torch.load(stage2_ckpt, map_location=device))
        model.eval()
        y_t, y_p, eids = _eval_fused(model, test_loader, device)
        df_gate = _attach_true_pred(df_test_base, y_t, y_p, target_cols)
        # df_test_base already contains expert_col; avoid inserting a duplicate.
        df_gate = _set_or_move_first(df_gate, expert_col, eids.astype(int))
        gate_pred_path = os.path.join(outdir, 'gate_test_predictions.csv')
        df_gate.to_csv(gate_pred_path, index=False)

        df_gate_metrics = _metrics_per_target(y_t, y_p, target_cols)
        df_gate_metrics.insert(0, 'stage', 'gate')
        gate_metrics_path = os.path.join(outdir, 'gate_test_metrics.csv')
        df_gate_metrics.to_csv(gate_metrics_path, index=False)
    else:
        # still create empty placeholders to keep downstream stable
        pd.DataFrame().to_csv(os.path.join(outdir, 'gate_test_predictions.csv'), index=False)
        pd.DataFrame().to_csv(os.path.join(outdir, 'gate_test_metrics.csv'), index=False)

    # -------------------- Finetune (joint) --------------------
    finetune_ckpt = os.path.join(ckpt_dir, 'best_finetune.pt')
    if not os.path.exists(finetune_ckpt):
        # fallback to canonical best_model.pt
        finetune_ckpt = os.path.join(ckpt_dir, 'best_model.pt')

    if os.path.exists(finetune_ckpt):
        model.load_state_dict(torch.load(finetune_ckpt, map_location=device))
        model.eval()
        y_t, y_p, eids = _eval_fused(model, test_loader, device)
        df_ft = _attach_true_pred(df_test_base, y_t, y_p, target_cols)
        # df_test_base already contains expert_col; avoid inserting a duplicate.
        df_ft = _set_or_move_first(df_ft, expert_col, eids.astype(int))
        ft_pred_path = os.path.join(outdir, 'finetune_test_predictions.csv')
        df_ft.to_csv(ft_pred_path, index=False)

        df_ft_metrics = _metrics_per_target(y_t, y_p, target_cols)
        df_ft_metrics.insert(0, 'stage', 'finetune')
        ft_metrics_path = os.path.join(outdir, 'finetune_test_metrics.csv')
        df_ft_metrics.to_csv(ft_metrics_path, index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(outdir, 'finetune_test_predictions.csv'), index=False)
        pd.DataFrame().to_csv(os.path.join(outdir, 'finetune_test_metrics.csv'), index=False)

    # -------------------- Pretrain experts (Stage1) --------------------
    rows_pred = []
    rows_metrics = []

    # We'll create one wide table by concatenating the expert-specific test rows.
    for eid in (1, 2, 3, 4):
        ckpt = os.path.join(ckpt_dir, f'best_stage1_expert{eid}.pt')
        if not os.path.exists(ckpt):
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        y_t, y_p, eids = _eval_expert(model, test_loader, device, eid=eid)
        if y_t.size == 0:
            continue

        # corresponding base df: filter test rows with this expert id
        mask = (df_test_base[expert_col].astype(int).values == int(eid))
        df_e_base = df_test_base.loc[mask].reset_index(drop=True)
        df_e = _attach_true_pred(df_e_base, y_t, y_p, target_cols)
        df_e.insert(0, 'expert_id', int(eid))
        rows_pred.append(df_e)

        df_m = _metrics_per_target(y_t, y_p, target_cols)
        df_m.insert(0, 'expert_id', int(eid))
        rows_metrics.append(df_m)

    pre_pred_path = os.path.join(outdir, 'pretrain_best_predictions.csv')
    if rows_pred:
        pd.concat(rows_pred, axis=0, ignore_index=True).to_csv(pre_pred_path, index=False)
    else:
        pd.DataFrame().to_csv(pre_pred_path, index=False)

    pre_metrics_path = os.path.join(outdir, 'pretrain_best_metrics.csv')
    if rows_metrics:
        pd.concat(rows_metrics, axis=0, ignore_index=True).to_csv(pre_metrics_path, index=False)
    else:
        pd.DataFrame().to_csv(pre_metrics_path, index=False)

    print(f"[export_results] Wrote exports to: {outdir}")


if __name__ == '__main__':
    main()
