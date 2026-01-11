import os
import sys
import argparse
import json
import traceback
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

from models.fusion_model import FusionModel
from utils.dataset import ZDataset
from utils.logger import get_file_logger


class FusionWrapper(torch.nn.Module):
    """Wrap FusionModel to return only fused predictions (B,T)."""
    def __init__(self, fusion_model: torch.nn.Module):
        super().__init__()
        self.fusion_model = fusion_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fusion_model(x)
        if isinstance(out, (tuple, list)):
            y = out[0]
        else:
            y = out['fused']
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return y


def _subset_to_x(subset, max_samples: int, device: str):
    xs = []
    limit = min(len(subset), int(max_samples))
    for i in range(limit):
        x, _, _ = subset[i]
        xs.append(x)
    X = torch.stack(xs, dim=0).to(device)
    return X


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"SHAP values must be 2D, got shape={arr.shape}")
    return arr


def _split_shap_values(vals, n_outputs: int) -> list[np.ndarray]:
    """Normalize SHAP output to list[(N,D)] of length n_outputs."""
    if isinstance(vals, list):
        out = [np.array(v) for v in vals]
        # sometimes DeepExplainer returns [N,D,T]
        if len(out) == 1 and n_outputs > 1:
            arr = np.array(out[0])
            if arr.ndim == 3 and arr.shape[-1] == n_outputs:
                out = [arr[:, :, j] for j in range(n_outputs)]
        return [_ensure_2d(v) for v in out]

    arr = np.array(vals)
    if arr.ndim == 2:
        return [_ensure_2d(arr)]
    if arr.ndim == 3:
        if arr.shape[-1] == n_outputs:
            return [_ensure_2d(arr[:, :, j]) for j in range(n_outputs)]
        if arr.shape[0] == n_outputs:
            return [_ensure_2d(arr[j, :, :]) for j in range(n_outputs)]
        if arr.shape[-1] == 1:
            return [_ensure_2d(arr[:, :, 0])]
    raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--background", type=int, default=600)
    parser.add_argument("--explain", type=int, default=600)
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "deep", "gradient"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = cfg["paths"]["save_dir"]
    shap_dir = os.path.join(save_dir, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    logger = get_file_logger(os.path.join(shap_dir, "shap_analysis.log"), name="shap")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {os.path.abspath(args.config)}")

    try:
        import shap
    except Exception:
        logger.exception("Failed to import shap. Please `pip install shap`.")
        raise

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.exception("Failed to import matplotlib. Please `pip install matplotlib`. ")
        raise

    # ---- dataset ----
    dataset = ZDataset(
        csv_path=cfg["paths"]["data"],
        scaler_path=cfg["paths"]["scaler"],
        cfg=cfg,
        train=False,
    )
    target_cols = list(getattr(dataset, "target_cols", []))
    feature_names = list(getattr(dataset, "feature_cols", []))

    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = int(dataset.input_dim)
    cfg["model"]["output_dim"] = int(getattr(dataset, "output_dim", len(target_cols) or 1))

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(42)
    train_set, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    model = FusionModel(cfg).to(device)
    ckpt_path = os.path.join(save_dir, "checkpoints", "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    wrapper = FusionWrapper(model).to(device)
    wrapper.eval()

    background_size = min(max(50, int(args.background)), len(train_set))
    explain_size = min(max(50, int(args.explain)), len(test_set))
    logger.info(f"background_size={background_size}, explain_size={explain_size}")

    X_bg = _subset_to_x(train_set, background_size, device)
    X_explain = _subset_to_x(test_set, explain_size, device)

    used = None
    shap_values_raw = None

    def run_deep():
        nonlocal used, shap_values_raw
        logger.info("Trying DeepExplainer(check_additivity=False) ...")
        explainer = shap.DeepExplainer(wrapper, X_bg)
        shap_values_raw = explainer.shap_values(X_explain, check_additivity=False)
        used = "DeepExplainer(check_additivity=False)"

    def run_grad():
        nonlocal used, shap_values_raw
        logger.info("Trying GradientExplainer ...")
        explainer = shap.GradientExplainer(wrapper, X_bg)
        shap_values_raw = explainer.shap_values(X_explain)
        used = "GradientExplainer"

    if args.method == "deep":
        run_deep()
    elif args.method == "gradient":
        run_grad()
    else:
        try:
            run_deep()
        except Exception as e:
            logger.warning(f"DeepExplainer failed, fallback to GradientExplainer. Err={repr(e)}")
            run_grad()

    if shap_values_raw is None:
        raise RuntimeError("SHAP failed: shap_values_raw is None")

    n_outputs = int(cfg["model"].get("output_dim", 1))
    if len(target_cols) != n_outputs:
        # 允许 target_cols 为空时的回退
        if not target_cols:
            target_cols = [f"target_{j}" for j in range(n_outputs)]

    shap_values_list = _split_shap_values(shap_values_raw, n_outputs)
    logger.info(f"Explainer used: {used}")
    logger.info(f"n_outputs={n_outputs}, shap shapes={[v.shape for v in shap_values_list]}")

    X_explain_np = X_explain.detach().cpu().numpy()

    # ---- per-target outputs ----
    artifacts = {
        "explainer": used,
        "targets": target_cols,
        "shap_dir": os.path.abspath(shap_dir),
        "summary_png": {},
        "bar_csv": {},
        "shap_csv": {},
    }

    for j, tgt in enumerate(target_cols):
        if j >= len(shap_values_list):
            break

        shap_values_arr = shap_values_list[j]
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(tgt))[:80]

        # summary plot
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_values_arr,
            features=X_explain_np,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        summary_png = os.path.join(shap_dir, f"shap_summary__{safe}.png")
        plt.savefig(summary_png, dpi=300)
        plt.close()

        # mean(|shap|) bar csv
        mean_abs = np.mean(np.abs(shap_values_arr), axis=0)
        df_bar = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
        )
        bar_csv = os.path.join(shap_dir, f"shap_bar__{safe}.csv")
        df_bar.to_csv(bar_csv, index=False)

        # shap values csv
        feat_df = pd.DataFrame(X_explain_np, columns=[f"feat::{c}" for c in feature_names])
        shap_df = pd.DataFrame(shap_values_arr, columns=[f"shap::{c}" for c in feature_names])
        out_df = pd.concat([feat_df, shap_df], axis=1)
        out_df["target"] = str(tgt)
        out_df["explainer"] = used
        shap_csv = os.path.join(shap_dir, f"shap_values__{safe}.csv")
        out_df.to_csv(shap_csv, index=False)

        artifacts["summary_png"][str(tgt)] = os.path.abspath(summary_png)
        artifacts["bar_csv"][str(tgt)] = os.path.abspath(bar_csv)
        artifacts["shap_csv"][str(tgt)] = os.path.abspath(shap_csv)

    # write shap artifacts json
    with open(os.path.join(shap_dir, "shap_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2, ensure_ascii=False)

    # update summary.json (optional)
    summary_path = os.path.join(save_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    summary.setdefault("ShapArtifacts", {})
    summary["ShapArtifacts"].update({
        "shap_artifacts_json": os.path.abspath(os.path.join(shap_dir, "shap_artifacts.json")),
    })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("SHAP analysis finished successfully.")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        # keep a short crash log next to shap outputs
        try:
            with open("shap_crash.log", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        raise
