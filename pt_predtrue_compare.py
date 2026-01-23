"""pt_predtrue_compare.py

Generate a single HTML that contains three **square** Pred-vs-True scatter plots:
  1) Pretrain (best expert checkpoints)
  2) Gate-only (best_stage2)
  3) Finetune (best_finetune / best_model)

Data sources are the clean exports produced by export_results.py:
  - pretrain_best_predictions.csv
  - gate_test_predictions.csv
  - finetune_test_predictions.csv

Each point is colored by phase/region id (expert_col, default: "no") with a
consistent palette across the 3 plots.

This script is intentionally "export-only": it never writes extra CSV.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Keep consistent with pt_viz.py
REGION_COLORS: Dict[int, str] = {1: "#E699A7", 2: "#FEDD9E", 3: "#A6D9C0", 4: "#71A7D2"}
REGION_LABELS: Dict[int, str] = {1: "气相区 (Gas)", 2: "液相区 (Liquid)", 3: "相变区 (Phase Change)", 4: "临界区 (Critical)"}


def _safe_slug(s: str, max_len: int = 80) -> str:
    s = str(s)
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in "-_+") else "_")
    slug = "".join(out).strip("_")
    return slug[:max_len] if slug else "target"


def _infer_target_cols(df: pd.DataFrame) -> List[str]:
    """Return list of target names like ["Z (-)", "phi (-)"] from y_true:: prefix."""
    targets = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith("y_true::"):
            t = c.split("::", 1)[1]
            if f"y_pred::{t}" in df.columns:
                targets.append(t)
    return targets


def _read_pred_file(path: str) -> Optional[pd.DataFrame]:
    if not path or (not os.path.exists(path)):
        return None
    try:
        df = pd.read_csv(path)
        if df.shape[0] == 0:
            return None
        return df
    except Exception:
        return None


def _downsample_stratified(df: pd.DataFrame, region_col: str, n_max: int, seed: int = 42) -> pd.DataFrame:
    """Downsample to at most n_max rows, approximately preserving region proportions."""
    if n_max is None or n_max <= 0:
        return df
    if len(df) <= n_max:
        return df

    rng = np.random.default_rng(seed)

    if region_col not in df.columns:
        return df.sample(n=n_max, random_state=seed)

    # compute allocation per region
    counts = df[region_col].value_counts(dropna=False)
    weights = counts.values.astype(float)
    weights = weights / max(weights.sum(), 1.0)
    alloc = np.floor(weights * n_max).astype(int)
    # distribute remaining
    rem = int(n_max - alloc.sum())
    if rem > 0:
        frac = (weights * n_max) - np.floor(weights * n_max)
        order = np.argsort(-frac)
        alloc[order[:rem]] += 1

    pieces = []
    for (region_val, _), k in zip(counts.items(), alloc):
        if k <= 0:
            continue
        sub = df[df[region_col] == region_val]
        if len(sub) <= k:
            pieces.append(sub)
        else:
            idx = rng.choice(sub.index.to_numpy(), size=k, replace=False)
            pieces.append(df.loc[idx])
    if not pieces:
        return df.sample(n=n_max, random_state=seed)
    out = pd.concat(pieces, axis=0, ignore_index=True)
    return out


def _get_xy(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
    yt_col = f"y_true::{target}"
    yp_col = f"y_pred::{target}"
    if yt_col not in df.columns or yp_col not in df.columns:
        raise ValueError(f"CSV missing columns for target '{target}': {yt_col} / {yp_col}")
    y_true = df[yt_col].to_numpy(dtype=float)
    y_pred = df[yp_col].to_numpy(dtype=float)
    return y_true, y_pred


def _global_range(items: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    vals = []
    for yt, yp in items:
        if yt.size:
            vals.append(yt)
        if yp.size:
            vals.append(yp)
    if not vals:
        return -1.0, 1.0
    allv = np.concatenate(vals)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(allv))
    vmax = float(np.max(allv))
    if abs(vmax - vmin) < 1e-12:
        pad = 1.0 if vmax == 0 else abs(vmax) * 0.05
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.03
    return vmin - pad, vmax + pad


def _add_stage_scatter(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    col: int,
    region_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    show_legend: bool,
):
    # Make sure region exists
    if region_col in df.columns:
        regions = df[region_col].astype(float).fillna(-1).astype(int).to_numpy()
    else:
        regions = np.full(len(df), -1, dtype=int)

    # one trace per region to get consistent legend + color
    uniq = list(dict.fromkeys(regions.tolist()))  # stable
    # prefer 1..4 ordering
    uniq = sorted(uniq)

    for rid in uniq:
        mask = regions == rid
        if not np.any(mask):
            continue
        color = REGION_COLORS.get(int(rid), "#999999")
        name = REGION_LABELS.get(int(rid), f"Region {rid}")
        fig.add_trace(
            go.Scattergl(
                x=y_true[mask],
                y=y_pred[mask],
                mode="markers",
                name=name,
                marker=dict(size=4, color=color, opacity=0.75),
                showlegend=show_legend,
                hovertemplate="True=%{x:.6g}<br>Pred=%{y:.6g}<br>Region=" + str(rid) + "<extra></extra>",
            ),
            row=row,
            col=col,
        )
        # only show legend once across subplots
        show_legend = False


def build_compare_html(
    *,
    exports_dir: str,
    out_html: str,
    target: str,
    region_col: str = "no",
    max_points: int = 50000,
    seed: int = 42,
):
    pre_path = os.path.join(exports_dir, "pretrain_best_predictions.csv")
    gate_path = os.path.join(exports_dir, "gate_test_predictions.csv")
    ft_path = os.path.join(exports_dir, "finetune_test_predictions.csv")

    df_pre = _read_pred_file(pre_path)
    df_gate = _read_pred_file(gate_path)
    df_ft = _read_pred_file(ft_path)

    if df_pre is None and df_gate is None and df_ft is None:
        raise RuntimeError(f"No usable prediction CSV found under: {exports_dir}")

    # Infer target if not present
    for df in [df_ft, df_gate, df_pre]:
        if df is not None:
            if target is None or target == "":
                ts = _infer_target_cols(df)
                if not ts:
                    raise ValueError("Cannot infer targets from prediction CSV (missing y_true:: / y_pred:: columns)")
                target = ts[0]
            break

    # Downsample each stage for HTML size
    if df_pre is not None:
        df_pre = _downsample_stratified(df_pre, region_col, max_points, seed=seed + 1)
    if df_gate is not None:
        df_gate = _downsample_stratified(df_gate, region_col, max_points, seed=seed + 2)
    if df_ft is not None:
        df_ft = _downsample_stratified(df_ft, region_col, max_points, seed=seed + 3)

    stage_items = []
    if df_pre is not None:
        stage_items.append(("预训练 (Pretrain)", df_pre))
    if df_gate is not None:
        stage_items.append(("门控训练 (Gate)", df_gate))
    if df_ft is not None:
        stage_items.append(("联合微调 (Finetune)", df_ft))

    # Prepare global axis range across stages
    xy_list = []
    for _, df in stage_items:
        yt, yp = _get_xy(df, target)
        xy_list.append((yt, yp))
    vmin, vmax = _global_range(xy_list)

    # Create 1x3 subplot (or fewer if missing stages)
    cols = len(stage_items)
    fig = make_subplots(rows=1, cols=cols, subplot_titles=[f"{name} | {target}" for name, _ in stage_items])

    for j, (name, df) in enumerate(stage_items, start=1):
        yt, yp = _get_xy(df, target)
        _add_stage_scatter(
            fig,
            df,
            row=1,
            col=j,
            region_col=region_col,
            y_true=yt,
            y_pred=yp,
            show_legend=(j == 1),
        )
        # diagonal line
        fig.add_trace(
            go.Scatter(
                x=[vmin, vmax],
                y=[vmin, vmax],
                mode="lines",
                line=dict(color="#333333", dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=j,
        )

        # Square axes per subplot
        # NOTE: Plotly's scaleanchor expects axis ids like 'x', 'x2', ... rather than
        # layout keys 'xaxis', 'xaxis2', ...
        xaxis_key = f"xaxis{'' if j == 1 else j}"
        yaxis_key = f"yaxis{'' if j == 1 else j}"
        xaxis_id = f"x{'' if j == 1 else j}"
        fig.layout[xaxis_key].update(title_text="True", range=[vmin, vmax])
        fig.layout[yaxis_key].update(title_text="Pred", range=[vmin, vmax], scaleanchor=xaxis_id, scaleratio=1)

    # Make each panel visually close to a square. (Exact pixel-square is hard with
    # subplot titles/legend/margins, but this keeps the plotting domains near-square.)
    side = 620
    fig.update_layout(
        title=f"Pred vs True (colored by phase) | {target}",
        height=side,
        width=side * cols,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=30, r=20, t=70, b=30),
    )

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports_dir", required=True, help="Path to <exp_dir>/exports")
    ap.add_argument("--out_html", default=None, help="Output html path (default: <exports_dir>/pred_true_compare__<target>.html)")
    ap.add_argument("--target", default="", help="Target name (e.g. 'Z (-)')")
    ap.add_argument("--expert_col", default="no", help="Phase/region column name (default: no)")
    ap.add_argument("--max_points", type=int, default=50000, help="Downsample cap per stage for HTML size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    target = args.target.strip()
    if args.out_html:
        out_html = args.out_html
    else:
        slug = _safe_slug(target) if target else "auto"
        out_html = os.path.join(args.exports_dir, f"pred_true_compare__{slug}.html")

    build_compare_html(
        exports_dir=args.exports_dir,
        out_html=out_html,
        target=target,
        region_col=args.expert_col,
        max_points=args.max_points,
        seed=args.seed,
    )
    print(f"[pt_predtrue_compare] Wrote: {out_html}")


if __name__ == "__main__":
    main()
