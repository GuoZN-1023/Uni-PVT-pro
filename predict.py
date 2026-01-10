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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/current.yaml",
        help="配置文件路径（需要和训练时一致）"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = cfg["paths"]["data"]
    scaler_path = cfg["paths"]["scaler"]
    target_col = cfg.get("target_col", "Z (-)")
    expert_col = cfg.get("expert_col", "no")
    subset_cfg = cfg.get("subset", None)

    save_dir = cfg["paths"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    region_dir = os.path.join(save_dir, "region_eval")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(region_dir, exist_ok=True)

    logger = get_file_logger(os.path.join(eval_dir, "eval.log"), name="eval")
    logger.info(f"Device: {device}")


    dataset = ZDataset(
        csv_path=data_path,
        scaler_path=scaler_path,
        train=False,
        target_col=target_col,
        expert_col=expert_col,
        subset_cfg=subset_cfg,
    )


    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = dataset.input_dim
    logger.info(
        f"Dataset input_dim = {dataset.input_dim}, "
        f"set cfg['model']['input_dim'] accordingly."
    )

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=g
    )

    batch_size = cfg["training"]["batch_size"]
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    full_ids = dataset.expert_ids
    test_indices = np.array(test_set.indices, dtype=int)


    df_test_states = None
    if hasattr(dataset, "num_df") and dataset.num_df is not None:
        try:
            df_test_states = dataset.num_df.iloc[test_indices].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Failed to slice dataset.num_df for test set: {e}")
    test_expert_ids = full_ids[test_indices]

    model = FusionModel(cfg).to(device)
    ckpt_path = os.path.join(save_dir, "checkpoints", "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise RuntimeError("Unexpected batch format in eval.")
            x = x.to(device)
            y = y.to(device)
            preds, _, _ = model(x)

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info(f"Test MAE={mae:.6f} MSE={mse:.6f} R2={r2:.6f}")

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "R2": float(r2),
        "N_test": int(len(y_true)),
    }
    with open(os.path.join(eval_dir, "metrics_summary.yaml"), "w") as f:
        yaml.dump(metrics, f)



# ---- prediction table + plots (multi-target) ----
target_cols = list(getattr(dataset, "target_cols", []))
if len(target_cols) == 0:
    target_cols = [cfg.get("target_col", "Z (-)")]

df_dict = {"expert_id": test_expert_ids.astype(int)}
for j, tgt in enumerate(target_cols):
    df_dict[f"y_true::{tgt}"] = y_true[:, j]
    df_dict[f"y_pred::{tgt}"] = y_pred[:, j]

df_pred = pd.DataFrame(df_dict)

pred_csv_path = os.path.join(eval_dir, "test_predictions.csv")
df_pred.to_csv(pred_csv_path, index=False)

# scatter plots per target
df_pred["expert_id_str"] = df_pred["expert_id"].astype(str)
color_map = {"1": "#E699A7", "2": "#FEDD9E", "3": "#A6D9C0", "4": "#71A7D2"}

scatter_paths = {}
for tgt in target_cols:
    xcol = f"y_true::{tgt}"
    ycol = f"y_pred::{tgt}"
    fig = px.scatter(
        df_pred,
        x=xcol,
        y=ycol,
        color="expert_id_str",
        color_discrete_map=color_map,
        title=f"True vs Pred: {tgt}",
        labels={xcol: f"True {tgt}", ycol: f"Pred {tgt}", "expert_id_str": "Expert ID (no)"},
    )
    min_val = float(min(df_pred[xcol].min(), df_pred[ycol].min()))
    max_val = float(max(df_pred[xcol].max(), df_pred[ycol].max()))
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(width=2, dash="dash"))

    safe = "".join(ch if ch.isalnum() else "_" for ch in str(tgt))[:80]
    scatter_path = os.path.join(eval_dir, f"true_vs_pred_scatter__{safe}.html")
    fig.write_html(scatter_path)
    scatter_paths[tgt] = os.path.abspath(scatter_path)

# region metrics (mean across targets + per-target columns)
region_metrics = []
for eid in sorted(df_pred["expert_id"].unique()):
    df_g = df_pred[df_pred["expert_id"] == eid]
    yt = np.stack([df_g[f"y_true::{t}"].values for t in target_cols], axis=1)
    yp = np.stack([df_g[f"y_pred::{t}"].values for t in target_cols], axis=1)
    met = compute_metrics(yt, yp)
    row = {
        "expert_id": int(eid),
        "n": int(met["n"]),
        "mae_mean": float(met["mae"]),
        "mse_mean": float(met["mse"]),
        "r2_mean": float(met["r2"]),
    }
    for j, tgt in enumerate(target_cols):
        row[f"mae::{tgt}"] = float(met["mae_per_target"][j])
        row[f"mse::{tgt}"] = float(met["mse_per_target"][j])
        row[f"r2::{tgt}"] = float(met["r2_per_target"][j])
    region_metrics.append(row)

region_metrics_df = pd.DataFrame(region_metrics)
region_metrics_csv = os.path.join(region_dir, "region_metrics.csv")
region_metrics_df.to_csv(region_metrics_csv, index=False)

# ---- update summary.json (optional) ----
summary_path = os.path.join(save_dir, "summary.json")
summary = {}
if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

summary.setdefault("EvalArtifacts", {})
summary["EvalArtifacts"].update({
    "metrics_summary": os.path.abspath(os.path.join(eval_dir, "metrics_summary.yaml")),
    "pred_csv": os.path.abspath(pred_csv_path),
    "scatter_html_map": scatter_paths,
})
summary.setdefault("RegionArtifacts", {})
summary["RegionArtifacts"].update({
    "region_metrics_csv": os.path.abspath(region_metrics_csv),
})

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

logger.info(f"Evaluation artifacts saved to {eval_dir} and {region_dir}")
logger.info(f"Summary updated at {summary_path}")


if __name__ == "__main__":
    main()