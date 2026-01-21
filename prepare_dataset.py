"""prepare_dataset.py

Build train/val/test datasets by sampling from many per-molecule CSV files.

What it does
  - Given a directory of per-molecule CSV files (e.g. 124 files), sample rows from each file.
  - Within each molecule, sampling is stratified by `expert_col` (e.g. 'no'), preserving
    phase composition.
  - The sampled rows from each molecule are split into train/val/test with the same
    per-phase proportions.
  - Outputs are saved as train.csv / val.csv / test.csv plus a manifest.json for reproducibility.

This script is designed to be called by run_all.py before training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _augment_df_compat_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-destructive alias/derived columns for compatibility.

    - Add unit-annotated aliases for compact columns (e.g. Z -> 'Z (-)').
    - Add compact aliases for unit-annotated columns (e.g. 'T_r (-)' -> T_r).
    - Derive absolute state columns when possible:
        (T_r, T_c) -> 'T (K)'
        (p_r, p_c) -> 'P (Pa)' and 'P (MPa)'

    This makes downstream training configs more robust when column naming differs
    across preprocessing pipelines.
    """
    df = df.copy()

    # Compact <-> unit-annotated aliases (reduced properties)
    alias_pairs = [
        ("T_r", "T_r (-)"),
        ("p_r", "p_r (-)"),
        ("omega", "omega (-)"),
        ("T_c", "T_c (K)"),
        ("p_c", "p_c (Pa)"),
        ("mu", "mu ((J*m^3/kmol)^0.5)"),
        ("Z", "Z (-)"),
        ("phi", "phi (-)"),
        ("lnphi", "lnphi (-)"),
        ("H", "H (J/mol)"),
        ("S", "S (J/mol/K)"),
        ("T", "T (K)"),
        ("P", "P (Pa)"),
    ]
    for a, b in alias_pairs:
        if a in df.columns and b not in df.columns:
            df[b] = df[a]
        if b in df.columns and a not in df.columns:
            df[a] = df[b]

    # Derive absolute state if possible
    if "T (K)" not in df.columns:
        if "T_r" in df.columns and "T_c" in df.columns:
            df["T (K)"] = pd.to_numeric(df["T_r"], errors="coerce") * pd.to_numeric(df["T_c"], errors="coerce")
        elif "T_r (-)" in df.columns and "T_c (K)" in df.columns:
            df["T (K)"] = pd.to_numeric(df["T_r (-)"], errors="coerce") * pd.to_numeric(df["T_c (K)"], errors="coerce")

    if ("P (Pa)" not in df.columns or "P (MPa)" not in df.columns):
        P_pa = None
        if "p_r" in df.columns and "p_c" in df.columns:
            P_pa = pd.to_numeric(df["p_r"], errors="coerce") * pd.to_numeric(df["p_c"], errors="coerce")
        elif "p_r (-)" in df.columns and "p_c (Pa)" in df.columns:
            P_pa = pd.to_numeric(df["p_r (-)"], errors="coerce") * pd.to_numeric(df["p_c (Pa)"], errors="coerce")
        if P_pa is not None:
            if "P (Pa)" not in df.columns:
                df["P (Pa)"] = P_pa
            if "P (MPa)" not in df.columns:
                df["P (MPa)"] = P_pa / 1.0e6

    # lnphi: derive from phi if needed
    if "lnphi (-)" not in df.columns:
        if "lnphi" in df.columns:
            df["lnphi (-)"] = df["lnphi"]
        elif "phi" in df.columns:
            phi = pd.to_numeric(df["phi"], errors="coerce")
            df["lnphi (-)"] = np.log(np.clip(phi, 1.0e-12, None))
        elif "phi (-)" in df.columns:
            phi = pd.to_numeric(df["phi (-)"], errors="coerce")
            df["lnphi (-)"] = np.log(np.clip(phi, 1.0e-12, None))

    return df


def _save_npz_table(df: pd.DataFrame, out_npz: Path) -> None:
    """Save a DataFrame as a compact NPZ that still supports flexible column selection.

    Format:
      - data: float32 matrix for all numeric columns (N, M)
      - __cols__: object array of numeric column names (M,)
      - obj__<col>: string array for each non-numeric column

    This allows training-time selection by column name without requiring CSV.
    """
    df = df.copy()

    # Normalize/augment columns so both compact and unit-annotated configs work.
    df = _augment_df_compat_names(df)

    # Keep numeric columns in a single matrix.
    num_df = df.select_dtypes(include=[np.number]).copy()
    # Coerce remaining numeric-looking cols (common when CSV is mixed dtype)
    for c in df.columns:
        if c in num_df.columns:
            continue
        # attempt numeric conversion if it looks like numbers with occasional strings
        if df[c].dtype == object:
            as_num = pd.to_numeric(df[c], errors="coerce")
            # treat as numeric if conversion yields mostly non-NaN
            if as_num.notna().mean() > 0.9:
                num_df[c] = as_num

    num_cols = list(num_df.columns)
    data = num_df.to_numpy(dtype=np.float32, copy=True)

    payload = {
        "data": data,
        "__cols__": np.asarray(num_cols, dtype=object),
    }

    # Store non-numeric columns separately (strings/ids).
    for c in df.columns:
        if c in num_cols:
            continue
        payload[f"obj__{c}"] = np.asarray(df[c].astype(str).fillna(""), dtype=object)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)


def _allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    """Allocate integer counts that sum to `total`, proportional to `weights`."""
    total = int(total)
    if total <= 0:
        return np.zeros_like(weights, dtype=int)

    w = np.asarray(weights, dtype=float)
    s = float(w.sum())
    if s <= 0 or len(w) == 0:
        out = np.zeros(len(w), dtype=int)
        if len(out) == 0:
            return out
        base = total // len(out)
        out[:] = base
        out[: total - base * len(out)] += 1
        return out

    raw = total * (w / s)
    flo = np.floor(raw).astype(int)
    rem = total - int(flo.sum())
    if rem > 0:
        frac = raw - flo
        idx = np.argsort(-frac)
        flo[idx[:rem]] += 1
    return flo


def _split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    ratios = ratios / ratios.sum()
    c = _allocate_counts(int(n), ratios)
    return int(c[0]), int(c[1]), int(c[2])


def _two_pass_molecule_targets(
    file_sizes: np.ndarray,
    total_rows: int,
) -> np.ndarray:
    """Allocate per-molecule target rows close to total_rows with a two-pass redistribution."""
    n_files = len(file_sizes)
    if n_files == 0:
        return np.array([], dtype=int)

    total_rows = int(total_rows)
    base = int(math.floor(total_rows / n_files))
    targets = np.minimum(file_sizes, base).astype(int)

    remaining = total_rows - int(targets.sum())
    if remaining <= 0:
        return targets

    capacity = (file_sizes - targets).clip(min=0)
    if int(capacity.sum()) <= 0:
        return targets

    add = _allocate_counts(remaining, capacity)
    targets = targets + np.minimum(add, capacity)
    return targets.astype(int)


def sample_one_molecule(
    csv_path: Path,
    *,
    expert_col: str,
    total_take: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(int(seed))
    df = pd.read_csv(csv_path)

    if expert_col not in df.columns:
        raise ValueError(f"Missing expert_col '{expert_col}' in {csv_path}")

    molecule_id = csv_path.stem
    df["molecule_id"] = molecule_id

    grp = df.groupby(expert_col, sort=True)
    no_values = np.array(list(grp.groups.keys()))
    sizes = np.array([len(grp.get_group(no)) for no in no_values], dtype=int)
    n_total = int(len(df))

    # Allocate per-no take proportionally (preserve phase composition).
    total_take = int(min(total_take, n_total))
    per_no_take = _allocate_counts(total_take, sizes)

    train_parts, val_parts, test_parts = [], [], []
    meta = {
        "molecule_id": molecule_id,
        "file": str(csv_path),
        "seed": int(seed),
        "n_total": int(n_total),
        "n_take": int(total_take),
        "by_no": {},
    }

    for no, take_n in zip(no_values, per_no_take):
        take_n = int(take_n)
        if take_n <= 0:
            continue

        sub = grp.get_group(no)
        take_n = min(take_n, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take_n, replace=False)
        rng.shuffle(idx)

        n_tr, n_va, n_te = _split_counts(take_n, train_ratio, val_ratio, test_ratio)
        idx_tr = idx[:n_tr]
        idx_va = idx[n_tr : n_tr + n_va]
        idx_te = idx[n_tr + n_va : n_tr + n_va + n_te]

        tr_df = df.loc[idx_tr].copy()
        va_df = df.loc[idx_va].copy()
        te_df = df.loc[idx_te].copy()

        tr_df["split"] = "train"
        va_df["split"] = "val"
        te_df["split"] = "test"

        train_parts.append(tr_df)
        val_parts.append(va_df)
        test_parts.append(te_df)

        meta["by_no"][str(no)] = {
            "take": int(take_n),
            "train": int(n_tr),
            "val": int(n_va),
            "test": int(n_te),
        }

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()

    meta["n_train"] = int(len(train_df))
    meta["n_val"] = int(len(val_df))
    meta["n_test"] = int(len(test_df))
    meta["n_out"] = int(len(train_df) + len(val_df) + len(test_df))
    return train_df, val_df, test_df, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules_dir", type=str, required=True, help="Directory containing per-molecule CSV files")
    ap.add_argument("--pattern", type=str, default="*.csv")
    ap.add_argument("--expert_col", type=str, default="no")
    ap.add_argument("--total_rows", type=int, default=1_000_000)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--export_npz", action="store_true", help="Also export train/val/test as .npz (and keep only npz if --delete_csv)")
    ap.add_argument("--delete_csv", action="store_true", help="Delete train/val/test.csv after exporting .npz")
    args = ap.parse_args()

    mol_dir = Path(args.molecules_dir)
    files = sorted(mol_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No CSV files found in {mol_dir} with pattern {args.pattern}")

    # File sizes for target allocation
    sizes = np.array([sum(1 for _ in open(f, "rb")) - 1 for f in files], dtype=int)
    # The above is line-count based (fast) and avoids reading whole CSVs.
    # Fallback if any file is empty or weird.
    sizes = np.maximum(sizes, 0)

    per_file_targets = _two_pass_molecule_targets(sizes, args.total_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    mol_seeds = rng.integers(0, 2**31 - 1, size=len(files), dtype=np.int64)

    train_all, val_all, test_all = [], [], []
    manifest = {
        "seed": int(args.seed),
        "molecules_dir": str(mol_dir),
        "pattern": args.pattern,
        "expert_col": args.expert_col,
        "total_rows_target": int(args.total_rows),
        "splits": {"train": float(args.train_ratio), "val": float(args.val_ratio), "test": float(args.test_ratio)},
        "molecules": [],
        "totals": {"train": 0, "val": 0, "test": 0, "all": 0},
    }

    for csv_path, target_n, s in zip(files, per_file_targets, mol_seeds):
        tr, va, te, meta = sample_one_molecule(
            csv_path,
            expert_col=args.expert_col,
            total_take=int(target_n),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(s),
        )
        train_all.append(tr)
        val_all.append(va)
        test_all.append(te)
        meta["target_take"] = int(target_n)
        manifest["molecules"].append(meta)

    train_df = pd.concat(train_all, ignore_index=True)
    val_df = pd.concat(val_all, ignore_index=True)
    test_df = pd.concat(test_all, ignore_index=True)

    manifest["totals"]["train"] = int(len(train_df))
    manifest["totals"]["val"] = int(len(val_df))
    manifest["totals"]["test"] = int(len(test_df))
    manifest["totals"]["all"] = int(len(train_df) + len(val_df) + len(test_df))

    train_path = outdir / "train.csv"
    val_path = outdir / "val.csv"
    test_path = outdir / "test.csv"
    manifest_path = outdir / "manifest.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Optional: export compact npz versions so training can run without CSV.
    if bool(args.export_npz):
        train_npz = outdir / "train.npz"
        val_npz = outdir / "val.npz"
        test_npz = outdir / "test.npz"
        _save_npz_table(train_df, train_npz)
        _save_npz_table(val_df, val_npz)
        _save_npz_table(test_df, test_npz)
        print(f"  train(npz): {train_npz}")
        print(f"  val  (npz): {val_npz}")
        print(f"  test (npz): {test_npz}")

        if bool(args.delete_csv):
            for p in [train_path, val_path, test_path]:
                try:
                    p.unlink(missing_ok=True)
                except TypeError:
                    # py<3.8 compat
                    if p.exists():
                        p.unlink()
            print("  deleted train/val/test.csv")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[prepare_dataset] saved")
    if train_path.exists():
        print(f"  train: {train_path} ({len(train_df)})")
    if val_path.exists():
        print(f"  val  : {val_path} ({len(val_df)})")
    if test_path.exists():
        print(f"  test : {test_path} ({len(test_df)})")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
