import os
import sys
import argparse
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
    """把 FusionModel 包一层，只返回 y_pred。"""
    def __init__(self, fusion_model: torch.nn.Module):
        super().__init__()
        self.fusion_model = fusion_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.fusion_model(x)

        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return y


def subset_to_xy(subset, max_samples=None, device="cpu"):

    xs, ys = [], []
    n = len(subset)
    limit = n if max_samples is None else min(n, max_samples)
    for i in range(limit):
        x, y, _ = subset[i]
        xs.append(x)
        y = y.view(-1)
        if y.numel() != 1:
            y = y[:1]
        ys.append(y)
    X = torch.stack(xs, dim=0).to(device)          
    y = torch.cat(ys, dim=0).to(device)            
    return X, y



def _ensure_2d_shap(arr: np.ndarray):
    arr = np.array(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"SHAP values must be 2D, got shape={arr.shape}")
    return arr


def _split_shap_values(vals, n_outputs: int):
    """Normalize SHAP output to list[(N,D)] of length n_outputs."""
    if isinstance(vals, list):
        out = [np.array(v) for v in vals]
        if len(out) == 1 and n_outputs > 1:
            arr = np.array(out[0])
            if arr.ndim == 3 and arr.shape[-1] == n_outputs:
                out = [arr[:, :, j] for j in range(n_outputs)]
        return [_ensure_2d_shap(v) for v in out]

    arr = np.array(vals)
    if arr.ndim == 2:
        return [_ensure_2d_shap(arr)]
    if arr.ndim == 3:
        if arr.shape[-1] == n_outputs:
            return [_ensure_2d_shap(arr[:, :, j]) for j in range(n_outputs)]
        if arr.shape[0] == n_outputs:
            return [_ensure_2d_shap(arr[j, :, :]) for j in range(n_outputs)]
        if arr.shape[-1] == 1:
            return [_ensure_2d_shap(arr[:, :, 0])]
    raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--background", type=int, default=800)
    parser.add_argument("--explain", type=int, default=800)
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
    logger.info(f"Save dir: {shap_dir}")


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
        logger.exception("Failed to import matplotlib. Please `pip install matplotlib`.")
        raise

    data_path = cfg["paths"]["data"]
    scaler_path = cfg["paths"]["scaler"]
    target_col = cfg.get("target_col", "Z (-)")
    expert_col = cfg.get("expert_col", "no")
    subset_cfg = cfg.get("subset", None)

    dataset = ZDataset(
        csv_path=data_path,
        scaler_path=scaler_path,
        cfg=cfg,
        train=False,
    )


    cfg.setdefault("model", {})
    cfg["model"]["input_dim"] = int(dataset.input_dim)
    feature_names = dataset.feature_cols
    logger.info(f"input_dim={dataset.input_dim}")
    logger.info(f"features({len(feature_names)}): {feature_names}")


    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(42)
    train_set, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)
    logger.info(f"Split: train={len(train_set)}, test={len(test_set)}")


    model = FusionModel(cfg).to(device)
    ckpt_path = os.path.join(save_dir, "checkpoints", "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    wrapper = FusionWrapper(model).to(device)
    wrapper.eval()

    background_size = min(max(50, args.background), len(train_set))
    explain_size = min(max(50, args.explain), len(test_set))
    logger.info(f"background_size={background_size}, explain_size={explain_size}")

    X_bg, _ = subset_to_xy(train_set, max_samples=background_size, device=device)
    X_explain, y_explain = subset_to_xy(test_set, max_samples=explain_size, device=device)

    shap_values_raw = None
    shap_values_list = None
    used = None

    def run_deep():
        nonlocal shap_values_raw, used
        logger.info("Trying DeepExplainer(check_additivity=False) ...")
        explainer = shap.DeepExplainer(wrapper, X_bg)
        vals = explainer.shap_values(X_explain, check_additivity=False)
        shap_values_raw = vals
        used = "DeepExplainer(check_additivity=False)"

    def run_grad():
        nonlocal shap_values_raw, used
        logger.info("Trying GradientExplainer ...")
        explainer = shap.GradientExplainer(wrapper, X_bg)
        vals = explainer.shap_values(X_explain)
        shap_values_raw = vals
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

    if shap_values_arr is None:
        raise RuntimeError("SHAP failed: shap_values_arr is None")


    X_explain_np = X_explain.detach().cpu().numpy()
    y_true_np = y_explain.detach().cpu().numpy().reshape(-1)
    with torch.no_grad():
        y_pred_np = wrapper(X_explain).detach().cpu().numpy().reshape(-1)

    target_cols = list(getattr(dataset, "target_cols", []))
    if len(target_cols) == 0:
        target_cols = [cfg.get("target_col", "Z (-)")]
    n_outputs = len(target_cols)
    shap_values_list = _split_shap_values(shap_values_raw, n_outputs)
    logger.info(f"Explainer used: {used}, n_outputs={n_outputs}, shap shapes={[v.shape for v in shap_values_list]}")



# ---- SHAP summary plots: one per target ----
for j, tgt in enumerate(target_cols):
    if j >= len(shap_values_list):
        break
    shap_values_arr = shap_values_list[j]
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(tgt))[:80]

    logger.info(f"Saving shap_summary__{safe}.png ...")
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

    # mean(|shap|) bar CSV per target
    mean_abs = np.mean(np.abs(shap_values_arr), axis=0)
    df_bar = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    bar_csv = os.path.join(shap_dir, f"shap_bar__{safe}.csv")
    df_bar.to_csv(bar_csv, index=False)



    logger.info("Saving shap_values.csv ...")
    feat_df = pd.DataFrame(X_explain_np, columns=[f"feat_{c}" for c in feature_names])
    shap_df = pd.DataFrame(shap_values_arr, columns=[f"shap_{c}" for c in feature_names])

    out_df = pd.concat([feat_df, shap_df], axis=1)
    out_df["y_true"] = y_true_np
    out_df["y_pred"] = y_pred_np
    out_df["explainer"] = used
    out_df.to_csv(os.path.join(shap_dir, "shap_values.csv"), index=False)

    logger.info("SHAP analysis finished successfully.")


if __name__ == "__main__":

    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        crash_log = "shap_crash.log"
        try:
            cfg_path = None
            if "--config" in sys.argv:
                cfg_path = sys.argv[sys.argv.index("--config") + 1]
            if cfg_path and os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                save_dir = cfg["paths"]["save_dir"]
                shap_dir = os.path.join(save_dir, "shap")
                os.makedirs(shap_dir, exist_ok=True)
                crash_log = os.path.join(shap_dir, "crash.log")
        except Exception:
            pass

        with open(crash_log, "a", encoding="utf-8") as f:
            f.write("\n========== SHAP FAILED ==========\n")
            f.write(traceback.format_exc())
            f.write("\n")
        sys.exit(1)
