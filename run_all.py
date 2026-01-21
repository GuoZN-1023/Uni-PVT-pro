# run_all.py
import os
import sys
import json
import shutil
import argparse
import traceback
from datetime import datetime
import yaml
import subprocess
import fnmatch
import csv as _csv

# Keep a global copy so the outer exception handler can still read config
BASE_CFG = {}


def _ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def _append(path: str, msg: str):
    if not path:
        return
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg if msg.endswith("\n") else msg + "\n")


def run_cmd(cmd, stdout_path: str, stderr_path: str, runall_log: str, cwd=None, allow_fail=False):
    _append(runall_log, f"\n[run_all] Running: {' '.join(cmd)}")
    _append(runall_log, f"[run_all]  stdout -> {stdout_path}")
    _append(runall_log, f"[run_all]  stderr -> {stderr_path}")

    _ensure_dir(os.path.dirname(stdout_path))
    _ensure_dir(os.path.dirname(stderr_path))

    with open(stdout_path, "a", encoding="utf-8") as out_f, open(stderr_path, "a", encoding="utf-8") as err_f:
        out_f.write(f"\n========== CMD START: {' '.join(cmd)} ==========\n")
        err_f.write(f"\n========== CMD START: {' '.join(cmd)} ==========\n")
        out_f.flush()
        err_f.flush()

        result = subprocess.run(cmd, cwd=cwd, stdout=out_f, stderr=err_f)

        out_f.write(f"========== CMD END (returncode={result.returncode}) ==========\n")
        err_f.write(f"========== CMD END (returncode={result.returncode}) ==========\n")
        out_f.flush()
        err_f.flush()

    _append(runall_log, f"[run_all] Return code: {result.returncode}")

    if result.returncode != 0 and not allow_fail:
        tail = ""
        try:
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail_lines = lines[-120:]
            tail = "".join(tail_lines)
        except Exception:
            tail = "(failed to read stderr log tail)"

        _append(runall_log, "[run_all] ---- stderr tail (last 120 lines) ----")
        _append(runall_log, tail)
        _append(runall_log, "[run_all] ---- end stderr tail ----")

        raise RuntimeError(
            f"Command failed with code {result.returncode}: {' '.join(cmd)}\n"
            f"See full logs:\n  stdout: {stdout_path}\n  stderr: {stderr_path}"
        )
    return result.returncode


def _safe_slug(s: str, max_len: int = 80) -> str:
    s = str(s)
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in "-_+") else "_")
    slug = "".join(out).strip("_")
    return slug[:max_len] if slug else "target"


def _infer_targets_from_pred_csv(pred_csv_path: str):
    """Infer target names from export csv header: y_true::xxx / y_pred::xxx"""
    try:
        with open(pred_csv_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip("\n")
        cols = [c.strip() for c in header.split(",") if c.strip()]
        y_true_cols = [c for c in cols if c.startswith("y_true::")]
        targets = []
        for c in y_true_cols:
            tgt = c.split("::", 1)[1]
            if f"y_pred::{tgt}" in cols:
                targets.append(tgt)
        return targets
    except Exception:
        return []


def _cleanup_extra_csv(exp_dir: str, keep_abs_paths: set[str], runall_log: str):
    """Delete all .csv under exp_dir except those in keep_abs_paths."""
    removed = 0
    for root, _, files in os.walk(exp_dir):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            p = os.path.abspath(os.path.join(root, fn))
            if p in keep_abs_paths:
                continue
            try:
                os.remove(p)
                removed += 1
            except Exception:
                pass
    _append(runall_log, f"[run_all] CSV cleanup removed {removed} extra csv files.")


def _build_mol_cache_input_csv(molecules_dir: str, out_csv: str, *, pattern: str, mol_id_col: str, smiles_col: str) -> int:
    """Scan per-molecule CSVs and write a compact (mol_id, SMILES) table.

    We only read header + first data row from each file to keep it cheap.
    Returns number of unique molecules collected.
    """
    mol_map = {}
    for root, _, files in os.walk(molecules_dir):
        for fn in files:
            if not fnmatch.fnmatch(fn, pattern):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = _csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue
                    header = [h.strip() for h in header]
                    if mol_id_col not in header or smiles_col not in header:
                        continue
                    mi = header.index(mol_id_col)
                    si = header.index(smiles_col)
                    row = None
                    for r in reader:
                        if r and any(x.strip() for x in r):
                            row = r
                            break
                    if row is None or mi >= len(row) or si >= len(row):
                        continue
                    mid = str(row[mi]).strip()
                    smi = str(row[si]).strip()
                    if not mid or not smi:
                        continue
                    mol_map.setdefault(mid, smi)
            except Exception:
                continue

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow([mol_id_col, smiles_col])
        for mid, smi in mol_map.items():
            w.writerow([mid, smi])

    return len(mol_map)


def _resolve_root_results_dir(base_cfg: dict) -> str:
    """
    A-mode directory scheme:
      exp_dir = <root_results_dir>/<timestamp>/

    Priority:
      1) paths.root_results
      2) paths.save_dir (if endswith /latest, use its parent)
      3) "results" (relative)
    """
    paths_cfg = base_cfg.get("paths", {}) or {}
    root = paths_cfg.get("root_results", None)

    if not root:
        save_dir = paths_cfg.get("save_dir", None)
        if save_dir:
            # If user gave a "latest" pointer path, treat its parent as root.
            base = os.path.basename(os.path.normpath(save_dir))
            if base.lower() == "latest":
                root = os.path.dirname(os.path.normpath(save_dir))
            else:
                root = save_dir

    if not root:
        root = "results"

    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--shap_strict", action="store_true", help="SHAP失败则整次任务失败（默认：SHAP失败只记录日志）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    global BASE_CFG
    BASE_CFG = base_cfg

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # A mode: exp_dir = root_results_dir/timestamp
    root_results_dir = _resolve_root_results_dir(base_cfg)
    exp_dir = os.path.join(root_results_dir, timestamp)

    logs_dir = os.path.join(exp_dir, "logs")
    _ensure_dir(exp_dir)
    _ensure_dir(logs_dir)
    _ensure_dir(os.path.join(exp_dir, "checkpoints"))

    runall_log = os.path.join(logs_dir, "run_all.log")
    _append(runall_log, "========== RUN_ALL START ==========")
    _append(runall_log, f"timestamp: {timestamp}")
    _append(runall_log, f"root_results_dir: {os.path.abspath(root_results_dir)}")
    _append(runall_log, f"exp_dir:    {os.path.abspath(exp_dir)}")
    _append(runall_log, f"base_config:{os.path.abspath(args.config)}")

    # Make a working cfg copy and force save_dir/scaler under exp_dir
    cfg = dict(base_cfg)
    cfg.setdefault("paths", {})
    cfg["paths"]["root_results"] = root_results_dir
    cfg["paths"]["save_dir"] = exp_dir
    cfg["paths"]["scaler"] = os.path.join(exp_dir, "scaler.pkl")

    exp_config_path = os.path.join(exp_dir, "config_used.yaml")
    with open(exp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    _ensure_dir("config")
    shutil.copy(exp_config_path, os.path.join("config", "current.yaml"))

    python_exec = sys.executable

    ds_out = os.path.join(logs_dir, "prepare_dataset.stdout.log")
    ds_err = os.path.join(logs_dir, "prepare_dataset.stderr.log")
    train_out = os.path.join(logs_dir, "train.stdout.log")
    train_err = os.path.join(logs_dir, "train.stderr.log")
    pred_out = os.path.join(logs_dir, "predict.stdout.log")
    pred_err = os.path.join(logs_dir, "predict.stderr.log")
    shap_out = os.path.join(logs_dir, "shap.stdout.log")
    shap_err = os.path.join(logs_dir, "shap.stderr.log")
    ptviz_out = os.path.join(logs_dir, "pt_viz.stdout.log")
    ptviz_err = os.path.join(logs_dir, "pt_viz.stderr.log")
    ptcmp_out = os.path.join(logs_dir, "pt_predtrue_compare.stdout.log")
    ptcmp_err = os.path.join(logs_dir, "pt_predtrue_compare.stderr.log")

    # ===== Optional: build sampled dataset from many per-molecule CSVs =====
    paths_cfg = cfg.get("paths", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    sampling_cfg = (data_cfg.get("sampling", {}) or {}) if isinstance(data_cfg, dict) else {}
    pattern = str(sampling_cfg.get("pattern", "*.csv"))

    molecules_dir = paths_cfg.get("molecules_dir", None)

    # Backward-compatible convenience:
    # If paths.data points to a DIRECTORY (containing many molecule CSVs), treat it as molecules_dir
    data_path = paths_cfg.get("data", None)
    if (not molecules_dir) and data_path and os.path.isdir(data_path):
        molecules_dir = data_path
        cfg.setdefault("paths", {})
        cfg["paths"]["molecules_dir"] = molecules_dir

    ds_enabled = bool(sampling_cfg.get("enabled", False)) or bool(molecules_dir)

    if ds_enabled:
        if not molecules_dir:
            raise ValueError(
                "Dataset sampling enabled but no molecules directory was provided. "
                "Set paths.molecules_dir to the folder containing molecule CSVs "
                "(or ensure paths.data points to a directory, not a single CSV)."
            )

        dataset_dir = os.path.join(exp_dir, "dataset")

        split_cfg = (data_cfg.get("split", {}) or {}) if isinstance(data_cfg, dict) else {}
        train_ratio = float(split_cfg.get("train_ratio", 0.8))
        val_ratio = float(split_cfg.get("val_ratio", 0.1))
        test_ratio = float(split_cfg.get("test_ratio", 0.1))
        seed = int((cfg.get("training", {}) or {}).get("seed", 42))

        total_rows = int(sampling_cfg.get("total_rows", 1_000_000))
        expert_col = str(cfg.get("expert_col", "no"))

        export_npz = bool(sampling_cfg.get("export_npz", True))
        delete_csv = bool(sampling_cfg.get("delete_csv", True))

        _append(runall_log, "\n===== STAGE: PREP_DATASET =====")
        cmd = [
                python_exec,
                "prepare_dataset.py",
                "--molecules_dir",
                str(molecules_dir),
                "--outdir",
                str(dataset_dir),
                "--total_rows",
                str(total_rows),
                "--expert_col",
                expert_col,
                "--pattern",
                pattern,
                "--train_ratio",
                str(train_ratio),
                "--val_ratio",
                str(val_ratio),
                "--test_ratio",
                str(test_ratio),
                "--seed",
                str(seed),
            ]

        if export_npz:
            cmd.append("--export_npz")
        if delete_csv:
            cmd.append("--delete_csv")

        run_cmd(
            cmd,
            stdout_path=ds_out,
            stderr_path=ds_err,
            runall_log=runall_log,
        )

        # Wire split files into config (train/predict/shap will consume them)
        cfg.setdefault("paths", {})
        ext = "npz" if export_npz else "csv"
        cfg["paths"]["train_data"] = os.path.join(dataset_dir, f"train.{ext}")
        cfg["paths"]["val_data"] = os.path.join(dataset_dir, f"val.{ext}")
        cfg["paths"]["test_data"] = os.path.join(dataset_dir, f"test.{ext}")

        # Persist updated config
        with open(exp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)
        shutil.copy(exp_config_path, os.path.join("config", "current.yaml"))

    # ===== Optional: build mol_cache (Uni-PVT 2.0) =====
    mol_cfg = cfg.get("mol_encoder", {}) or {}
    mol_enabled = bool(mol_cfg.get("enabled", False))
    if mol_enabled:
        cache_cfg = mol_cfg.get("cache", {}) or {}
        cache_dir = cache_cfg.get("dir", os.path.join(exp_dir, "mol_cache"))
        z_conf_path = cache_cfg.get("z_conf_path", os.path.join(cache_dir, "z_conf.npy"))
        e_conf_path = cache_cfg.get("e_conf_path", os.path.join(cache_dir, "e_conf.npy"))
        id_map_path = cache_cfg.get("id_map_path", os.path.join(cache_dir, "mol_index.csv"))
        meta_path = cache_cfg.get("meta_path", os.path.join(cache_dir, "meta.json"))

        need_build = not (os.path.exists(z_conf_path) and os.path.exists(e_conf_path) and os.path.exists(id_map_path) and os.path.exists(meta_path))
        _append(runall_log, f"\n===== STAGE: BUILD_MOL_CACHE (enabled=True, need_build={need_build}) =====")

        if need_build:
            # Build a compact mol list CSV (mol_id, SMILES) from per-molecule CSV files
            smiles_col = str(mol_cfg.get("smiles_col", "SMILES"))
            mol_id_col = str(mol_cfg.get("mol_id_col", "mol_id"))
            cache_input = os.path.join(exp_dir, "mol_cache_input.csv")
            n_mols = _build_mol_cache_input_csv(
                molecules_dir if molecules_dir else paths_cfg.get("data", ""),
                cache_input,
                pattern=pattern,
                mol_id_col=mol_id_col,
                smiles_col=smiles_col,
            )
            if n_mols == 0:
                raise RuntimeError(
                    "mol_encoder.enabled=true but failed to build a mol_cache input table. "
                    f"Please ensure your per-molecule CSVs contain columns '{mol_id_col}' and '{smiles_col}'."
                )

            _ensure_dir(cache_dir)
            molcache_out = os.path.join(logs_dir, "build_mol_cache.stdout.log")
            molcache_err = os.path.join(logs_dir, "build_mol_cache.stderr.log")
            run_cmd(
                [python_exec, "build_mol_cache.py", "--csv", cache_input, "--outdir", cache_dir,
                 "--smiles_col", smiles_col, "--mol_id_col", mol_id_col],
                stdout_path=molcache_out,
                stderr_path=molcache_err,
                runall_log=runall_log,
            )
            # Keep workspace clean; dataset output is still only train/val/test.*
            try:
                os.remove(cache_input)
            except Exception:
                pass

    _append(runall_log, "\n===== STAGE: TRAIN =====")
    run_cmd(
        [python_exec, "train.py", "--config", exp_config_path],
        stdout_path=train_out,
        stderr_path=train_err,
        runall_log=runall_log,
    )

    _append(runall_log, "\n===== STAGE: EXPORT_RESULTS =====")
    run_cmd(
        [python_exec, "export_results.py", "--config", exp_config_path, "--outdir", os.path.join(exp_dir, "exports")],
        stdout_path=pred_out,
        stderr_path=pred_err,
        runall_log=runall_log,
    )

    shap_cfg = cfg.get("shap", {}) or {}
    shap_enabled = bool(shap_cfg.get("enabled", False))
    _append(runall_log, f"\n===== STAGE: SHAP (enabled={shap_enabled}, strict={args.shap_strict}) =====")

    shap_rc = None
    if shap_enabled:
        shap_rc = run_cmd(
            [python_exec, "shap_analysis.py", "--config", exp_config_path],
            stdout_path=shap_out,
            stderr_path=shap_err,
            runall_log=runall_log,
            allow_fail=(not args.shap_strict),
        )
        if shap_rc != 0:
            _append(
                runall_log,
                f"[run_all] SHAP failed (rc={shap_rc}). See:\n  {shap_err}\n  {os.path.join(exp_dir, 'shap', 'crash.log')}",
            )
    else:
        _append(runall_log, "[run_all] SHAP disabled. Set shap.enabled: true in config to enable.")

    # ---- PT visualization (HTML dashboard) ----
    ptviz_cfg = cfg.get("pt_viz", {}) or {}
    ptviz_enabled = bool(ptviz_cfg.get("enabled", True))
    ptviz_allow_fail = bool(ptviz_cfg.get("allow_fail", True))
    ptviz_diff_mode = str(ptviz_cfg.get("diff_mode", "pred_minus_true"))

    _append(
        runall_log,
        f"\n===== STAGE: PT_VIZ (enabled={ptviz_enabled}, allow_fail={ptviz_allow_fail}, diff_mode={ptviz_diff_mode}) =====",
    )

    ptviz_html_map = {}
    ptcmp_html_map = {}

    if ptviz_enabled:
        pred_csv = os.path.join(exp_dir, "exports", "finetune_test_predictions.csv")
        if os.path.exists(pred_csv):
            targets = _infer_targets_from_pred_csv(pred_csv)
            if targets:
                for tgt in targets:
                    slug = _safe_slug(tgt)
                    outdir = os.path.join(exp_dir, "pt_viz", slug)
                    rc = run_cmd(
                        [python_exec, "pt_viz.py", "--csv", pred_csv, "--outdir", outdir, "--target", tgt, "--diff_mode", ptviz_diff_mode],
                        stdout_path=ptviz_out,
                        stderr_path=ptviz_err,
                        runall_log=runall_log,
                        allow_fail=ptviz_allow_fail,
                    )
                    if rc == 0:
                        html = os.path.join(outdir, "pt_viz_dashboard.html")
                        if os.path.exists(html):
                            ptviz_html_map[tgt] = os.path.abspath(html)

                    exports_dir = os.path.join(exp_dir, "exports")
                    cmp_html = os.path.join(exp_dir, "pt_viz", f"pred_true_compare__{slug}.html")
                    expert_col = str(cfg.get("expert_col", "no"))
                    rc2 = run_cmd(
                        [python_exec, "pt_predtrue_compare.py", "--exports_dir", exports_dir, "--out_html", cmp_html, "--target", tgt, "--expert_col", expert_col],
                        stdout_path=ptcmp_out,
                        stderr_path=ptcmp_err,
                        runall_log=runall_log,
                        allow_fail=True,
                    )
                    if rc2 == 0 and os.path.exists(cmp_html):
                        ptcmp_html_map[tgt] = os.path.abspath(cmp_html)
            else:
                outdir = os.path.join(exp_dir, "pt_viz")
                rc = run_cmd(
                    [python_exec, "pt_viz.py", "--csv", pred_csv, "--outdir", outdir, "--diff_mode", ptviz_diff_mode],
                    stdout_path=ptviz_out,
                    stderr_path=ptviz_err,
                    runall_log=runall_log,
                    allow_fail=ptviz_allow_fail,
                )
                if rc == 0:
                    html = os.path.join(outdir, "pt_viz_dashboard.html")
                    if os.path.exists(html):
                        ptviz_html_map["__single__"] = os.path.abspath(html)

                exports_dir = os.path.join(exp_dir, "exports")
                cmp_html = os.path.join(exp_dir, "pt_viz", "pred_true_compare__single.html")
                expert_col = str(cfg.get("expert_col", "no"))
                rc2 = run_cmd(
                    [python_exec, "pt_predtrue_compare.py", "--exports_dir", exports_dir, "--out_html", cmp_html, "--expert_col", expert_col],
                    stdout_path=ptcmp_out,
                    stderr_path=ptcmp_err,
                    runall_log=runall_log,
                    allow_fail=True,
                )
                if rc2 == 0 and os.path.exists(cmp_html):
                    ptcmp_html_map["__single__"] = os.path.abspath(cmp_html)
        else:
            _append(runall_log, f"[run_all] PT_VIZ skipped: prediction csv not found: {pred_csv}")
    else:
        _append(runall_log, "[run_all] PT_VIZ disabled. Set pt_viz.enabled: true in config to enable.")

    # ---- Summary update ----
    summary_path = os.path.join(exp_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    summary.setdefault("RunAll", {})
    summary["RunAll"].update(
        {
            "timestamp": timestamp,
            "root_results_dir": os.path.abspath(root_results_dir),
            "exp_dir": os.path.abspath(exp_dir),
            "config_used": os.path.abspath(exp_config_path),
            "run_all_log": os.path.abspath(runall_log),
            "shap_enabled": shap_enabled,
            "shap_returncode": shap_rc,
            "logs": {
                "prepare_dataset_stdout": os.path.abspath(ds_out),
                "prepare_dataset_stderr": os.path.abspath(ds_err),
                "train_stdout": os.path.abspath(train_out),
                "train_stderr": os.path.abspath(train_err),
                "export_stdout": os.path.abspath(pred_out),
                "export_stderr": os.path.abspath(pred_err),
                "shap_stdout": os.path.abspath(shap_out),
                "shap_stderr": os.path.abspath(shap_err),
                "pt_viz_stdout": os.path.abspath(ptviz_out),
                "pt_viz_stderr": os.path.abspath(ptviz_err),
                "pt_predtrue_compare_stdout": os.path.abspath(ptcmp_out),
                "pt_predtrue_compare_stderr": os.path.abspath(ptcmp_err),
            },
        }
    )

    # ---- Keep only the 6 export CSVs; remove any other CSV artifacts created by submodules. ----
    exports_dir = os.path.join(exp_dir, "exports")
    keep_names = [
        "pretrain_best_predictions.csv",
        "pretrain_best_metrics.csv",
        "gate_test_predictions.csv",
        "gate_test_metrics.csv",
        "finetune_test_predictions.csv",
        "finetune_test_metrics.csv",
    ]
    keep_abs = set(os.path.abspath(os.path.join(exports_dir, name)) for name in keep_names)
    _cleanup_extra_csv(exp_dir, keep_abs_paths=keep_abs, runall_log=runall_log)

    summary.setdefault("PtVizArtifacts", {})
    summary["PtVizArtifacts"].update({"dashboards": ptviz_html_map, "pred_true_compare": ptcmp_html_map})

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _append(runall_log, "\n✅ RUN_ALL FINISHED")
    _append(runall_log, f"summary.json: {os.path.abspath(summary_path)}")
    _append(runall_log, "========== RUN_ALL END ==========")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()

        # Try best-effort logging even when crash happens early.
        try:
            root = _resolve_root_results_dir(BASE_CFG) if isinstance(BASE_CFG, dict) else "results"
            candidate = os.path.join(root, "run_all_error.log")

            if os.path.isdir(root):
                # If there are subdirs, try write into the latest run's logs
                subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
                if subdirs:
                    candidate = os.path.join(root, subdirs[-1], "logs", "run_all.log")

            _append(candidate, "\n❌ RUN_ALL FAILED")
            _append(candidate, tb)
        except Exception:
            pass

        sys.exit(1)