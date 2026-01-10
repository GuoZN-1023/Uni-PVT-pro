import os
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

from models.fusion_model import FusionModel
from utils.dataset import ZDataset
from utils.logger import get_file_logger


def _safe_r2(y_true_1d: np.ndarray, y_pred_1d: np.ndarray) -> float:
    try:
        return float(r2_score(y_true_1d, y_pred_1d))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true/y_pred: (N,T)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")

    T = y_true.shape[1]
    mae_t, mse_t, r2_t = [], [], []
    for j in range(T):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        mae_t.append(float(mean_absolute_error(yt, yp)))
        mse_t.append(float(mean_squared_error(yt, yp)))
        r2_t.append(_safe_r2(yt, yp))

    mae_mean = float(np.nanmean(mae_t))
    mse_mean = float(np.nanmean(mse_t))
    r2_mean = float(np.nanmean(r2_t))
    return {
        "n": int(y_true.shape[0]),
        "mae_mean": mae_mean,
        "mse_mean": mse_mean,
        "r2_mean": r2_mean,
        "mae_per_target": mae_t,
        "mse_per_target": mse_t,
        "r2_per_target": r2_t,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/current.yaml",
        help="配置文件路径（需要和训练时一致）",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = cfg["paths"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    region_dir = os.path.join(save_dir, "region_eval")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(region_dir, exist_ok=True)

    logger = get_file_logger(os.path.join(eval_dir, "eval.log"), name="eval")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {os.path.abspath(args.config)}")

    # ---- dataset (train=False: reuse scaler) ----
    data_path = cfg["paths"]["data"]
    scaler_path = cfg["paths"]["scaler"]

    dataset = ZDataset(
        csv_path=data_path,
        scaler_path=scaler_path,
        cfg=cfg,
        train=False,
    )

    target_cols = list(getattr(dataset, "target_cols", []))
    feature_cols = list(getattr(dataset, "feature_cols", []))
    expert_col = str(getattr(dataset, "expert_col", cfg.get("expert_col", "no")))
    logger.info(f"features({len(feature_cols)}): {feature_cols}")
    logger.info(f"targets({len(target_cols)}): {target_cols}")
    logger.info(f"expert_col: {expert_col}")

    # ---- model dims must match ----
    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = int(dataset.input_dim)
    cfg["model"]["output_dim"] = int(getattr(dataset, "output_dim", len(target_cols) or 1))

    # ---- split (must match train.py seed/ratios) ----
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(42)
    _, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    batch_size = int(cfg["training"]["batch_size"])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    test_indices = np.array(test_set.indices, dtype=int)
    test_expert_ids = np.asarray(dataset.expert_ids)[test_indices].astype(int)

    df_test_states = None
    if hasattr(dataset, "num_df") and dataset.num_df is not None:
        try:
            df_test_states = dataset.num_df.iloc[test_indices].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Failed to slice dataset.num_df for test set: {e}")

    # ---- load checkpoint ----
    model = FusionModel(cfg).to(device)
    ckpt_path = os.path.join(save_dir, "checkpoints", "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true_list, fused_list, w_list, experts_list = [], [], [], []

    with torch.no_grad():
        for x, y, _eid in test_loader:
            x = x.to(device)
            y = y.to(device)
            fused, w, expert_outputs = model(x)
            y_true_list.append(y.detach().cpu().numpy())
            fused_list.append(fused.detach().cpu().numpy())
            w_list.append(w.detach().cpu().numpy())
            experts_list.append(expert_outputs.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)              # (N,T)
    y_fused = np.concatenate(fused_list, axis=0)              # (N,T)
    gate_w = np.concatenate(w_list, axis=0)                   # (N,4)
    y_experts = np.concatenate(experts_list, axis=0)          # (N,4,T)

    # ---- fused metrics (overall) ----
    fused_met = compute_metrics(y_true, y_fused)
    logger.info(
        f"Fused metrics: MAE_mean={fused_met['mae_mean']:.6f} "
        f"MSE_mean={fused_met['mse_mean']:.6f} R2_mean={fused_met['r2_mean']:.6f}"
    )

    # 为兼容旧版：metrics_summary.yaml 里保留 MAE/MSE/R2 三个键
    metrics_summary = {
        "MAE": float(fused_met["mae_mean"]),
        "MSE": float(fused_met["mse_mean"]),
        "R2": float(fused_met["r2_mean"]),
        "N_test": int(fused_met["n"]),
        "targets": target_cols,
        "MAE_per_target": fused_met["mae_per_target"],
        "MSE_per_target": fused_met["mse_per_target"],
        "R2_per_target": fused_met["r2_per_target"],
    }
    with open(os.path.join(eval_dir, "metrics_summary.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(metrics_summary, f, allow_unicode=True)

    # ---- expert-only metrics (overall) ----
    expert_names = ["gas", "liquid", "critical", "extra"]
    expert_metrics_rows = []
    for k, name in enumerate(expert_names):
        met = compute_metrics(y_true, y_experts[:, k, :])
        row = {
            "expert_model": name,
            "n": int(met["n"]),
            "mae_mean": float(met["mae_mean"]),
            "mse_mean": float(met["mse_mean"]),
            "r2_mean": float(met["r2_mean"]),
        }
        for j, tgt in enumerate(target_cols):
            row[f"mae::{tgt}"] = float(met["mae_per_target"][j])
            row[f"mse::{tgt}"] = float(met["mse_per_target"][j])
            row[f"r2::{tgt}"] = float(met["r2_per_target"][j])
        expert_metrics_rows.append(row)

    expert_metrics_df = pd.DataFrame(expert_metrics_rows)
    expert_metrics_csv = os.path.join(eval_dir, "expert_metrics.csv")
    expert_metrics_df.to_csv(expert_metrics_csv, index=False)

    # ---- prediction table (keep old columns when single-target) ----
    if df_test_states is not None:
        df_pred = df_test_states.copy().reset_index(drop=True)
    else:
        df_pred = pd.DataFrame(index=np.arange(len(y_true)))

    df_pred["expert_id"] = test_expert_ids.astype(int)

    # gate weights
    df_pred["gate_w_gas"] = gate_w[:, 0]
    df_pred["gate_w_liquid"] = gate_w[:, 1]
    df_pred["gate_w_critical"] = gate_w[:, 2]
    df_pred["gate_w_extra"] = gate_w[:, 3]

    # per-target columns
    for j, tgt in enumerate(target_cols):
        df_pred[f"y_true::{tgt}"] = y_true[:, j]
        df_pred[f"y_pred::{tgt}"] = y_fused[:, j]
        df_pred[f"y_pred_gas::{tgt}"] = y_experts[:, 0, j]
        df_pred[f"y_pred_liquid::{tgt}"] = y_experts[:, 1, j]
        df_pred[f"y_pred_critical::{tgt}"] = y_experts[:, 2, j]
        df_pred[f"y_pred_extra::{tgt}"] = y_experts[:, 3, j]

        # 可选：硬路由（若 expert_id 在 1..4 范围）
        hard = np.full((len(df_pred),), np.nan, dtype=float)
        mask = (df_pred["expert_id"].values >= 1) & (df_pred["expert_id"].values <= 4)
        idx = df_pred.loc[mask, "expert_id"].values.astype(int) - 1
        hard[mask] = y_experts[mask, idx, j]
        df_pred[f"y_pred_hard::{tgt}"] = hard

    # 兼容旧版：单目标时保留 y_true/y_pred
    if len(target_cols) == 1:
        only = target_cols[0]
        df_pred["y_true"] = df_pred[f"y_true::{only}"]
        df_pred["y_pred"] = df_pred[f"y_pred::{only}"]

    pred_csv_path = os.path.join(eval_dir, "test_predictions.csv")
    df_pred.to_csv(pred_csv_path, index=False)

    # ---- scatter plots: per target, fused output ----
    scatter_paths = {}
    df_plot = df_pred.copy()
    df_plot["expert_id_str"] = df_plot["expert_id"].astype(str)
    color_map = {"1": "#E699A7", "2": "#FEDD9E", "3": "#A6D9C0", "4": "#71A7D2"}
    for tgt in target_cols:
        xcol = f"y_true::{tgt}"
        ycol = f"y_pred::{tgt}"
        fig = px.scatter(
            df_plot,
            x=xcol,
            y=ycol,
            color="expert_id_str",
            color_discrete_map=color_map,
            title=f"True vs Pred (Fused): {tgt}",
            labels={xcol: f"True {tgt}", ycol: f"Pred {tgt}", "expert_id_str": "Expert ID (no)"},
        )
        min_val = float(min(df_plot[xcol].min(), df_plot[ycol].min()))
        max_val = float(max(df_plot[xcol].max(), df_plot[ycol].max()))
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(dash="dash"))
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(tgt))[:80]
        out = os.path.join(eval_dir, f"true_vs_pred_scatter__{safe}.html")
        fig.write_html(out)
        scatter_paths[tgt] = os.path.abspath(out)

    # 兼容旧版：给第一目标再写一份固定文件名 true_vs_pred_scatter.html
    if target_cols:
        first_tgt = target_cols[0]
        legacy_path = os.path.join(eval_dir, "true_vs_pred_scatter.html")
        try:
            # 直接复制生成的 html（避免重复画图）
            import shutil
            shutil.copyfile(scatter_paths[first_tgt], legacy_path)
            scatter_paths["__legacy__"] = os.path.abspath(legacy_path)
        except Exception:
            pass

    # ---- region metrics: by expert_id (no) ----
    region_rows_fused = []
    region_rows_experts = []
    region_rows_hard = []

    for eid in sorted(df_pred["expert_id"].unique()):
        mask = (df_pred["expert_id"].values == eid)
        if mask.sum() == 0:
            continue

        yt = y_true[mask, :]
        yf = y_fused[mask, :]
        met_f = compute_metrics(yt, yf)
        row_f = {
            "expert_id": int(eid),
            "n": int(met_f["n"]),
            "mae_mean": float(met_f["mae_mean"]),
            "mse_mean": float(met_f["mse_mean"]),
            "r2_mean": float(met_f["r2_mean"]),
        }
        for j, tgt in enumerate(target_cols):
            row_f[f"mae::{tgt}"] = float(met_f["mae_per_target"][j])
            row_f[f"mse::{tgt}"] = float(met_f["mse_per_target"][j])
            row_f[f"r2::{tgt}"] = float(met_f["r2_per_target"][j])
        region_rows_fused.append(row_f)

        # 每个专家网络在该区域上的表现
        for k, name in enumerate(expert_names):
            met_e = compute_metrics(yt, y_experts[mask, k, :])
            row_e = {
                "expert_id": int(eid),
                "expert_model": name,
                "n": int(met_e["n"]),
                "mae_mean": float(met_e["mae_mean"]),
                "mse_mean": float(met_e["mse_mean"]),
                "r2_mean": float(met_e["r2_mean"]),
            }
            for j, tgt in enumerate(target_cols):
                row_e[f"mae::{tgt}"] = float(met_e["mae_per_target"][j])
                row_e[f"mse::{tgt}"] = float(met_e["mse_per_target"][j])
                row_e[f"r2::{tgt}"] = float(met_e["r2_per_target"][j])
            region_rows_experts.append(row_e)

        # 硬路由（如果 eid=1..4 才有意义）
        if 1 <= int(eid) <= 4:
            met_h = compute_metrics(yt, y_experts[mask, int(eid) - 1, :])
            row_h = {
                "expert_id": int(eid),
                "router": "hard_by_no",
                "n": int(met_h["n"]),
                "mae_mean": float(met_h["mae_mean"]),
                "mse_mean": float(met_h["mse_mean"]),
                "r2_mean": float(met_h["r2_mean"]),
            }
            for j, tgt in enumerate(target_cols):
                row_h[f"mae::{tgt}"] = float(met_h["mae_per_target"][j])
                row_h[f"mse::{tgt}"] = float(met_h["mse_per_target"][j])
                row_h[f"r2::{tgt}"] = float(met_h["r2_per_target"][j])
            region_rows_hard.append(row_h)

    region_metrics_fused_csv = os.path.join(region_dir, "region_metrics.csv")
    pd.DataFrame(region_rows_fused).to_csv(region_metrics_fused_csv, index=False)

    # 兼容旧版：同时输出一个 yaml 方便人读
    try:
        region_metrics_yaml = os.path.join(region_dir, "region_metrics.yaml")
        with open(region_metrics_yaml, "w", encoding="utf-8") as f:
            yaml.dump({"regions": region_rows_fused}, f, allow_unicode=True)
    except Exception:
        region_metrics_yaml = ""

    region_metrics_experts_csv = os.path.join(region_dir, "region_expert_metrics.csv")
    pd.DataFrame(region_rows_experts).to_csv(region_metrics_experts_csv, index=False)

    region_metrics_hard_csv = os.path.join(region_dir, "region_hard_router_metrics.csv")
    pd.DataFrame(region_rows_hard).to_csv(region_metrics_hard_csv, index=False)

    # ---- update summary.json (keep old keys, add new ones) ----
    summary_path = os.path.join(save_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    summary.setdefault("EvalArtifacts", {})
    summary["EvalArtifacts"].update({
        "metrics_summary": os.path.abspath(os.path.join(eval_dir, "metrics_summary.yaml")),
        "pred_csv": os.path.abspath(pred_csv_path),
        "expert_metrics_csv": os.path.abspath(expert_metrics_csv),
        "scatter_html_map": scatter_paths,
    })
    summary.setdefault("RegionArtifacts", {})
    summary["RegionArtifacts"].update({
        "region_metrics_csv": os.path.abspath(region_metrics_fused_csv),
        "region_metrics_yaml": os.path.abspath(region_metrics_yaml) if region_metrics_yaml else "",
        "region_expert_metrics_csv": os.path.abspath(region_metrics_experts_csv),
        "region_hard_router_metrics_csv": os.path.abspath(region_metrics_hard_csv),
    })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation artifacts saved to {eval_dir} and {region_dir}")


if __name__ == "__main__":
    main()
