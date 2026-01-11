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


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _append(path: str, msg: str):
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
        out_f.flush(); err_f.flush()

        result = subprocess.run(cmd, cwd=cwd, stdout=out_f, stderr=err_f)

        out_f.write(f"========== CMD END (returncode={result.returncode}) ==========\n")
        err_f.write(f"========== CMD END (returncode={result.returncode}) ==========\n")
        out_f.flush(); err_f.flush()

    _append(runall_log, f"[run_all] Return code: {result.returncode}")

    if result.returncode != 0 and not allow_fail:
        # Surface the last part of stderr to make debugging easier.
        tail = ""
        try:
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail_lines = lines[-120:]  # last ~120 lines
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
    """从 predict.py 输出的 test_predictions.csv 里读取目标名列表（y_true::xxx / y_pred::xxx）。"""
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
    """Delete all .csv under exp_dir except the ones in keep_abs_paths.
    Keep plots/html/images/logs/etc. This is to keep outputs tidy.
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--shap_strict", action="store_true", help="SHAP失败则整次任务失败（默认：SHAP失败只记录日志）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_results_dir = base_cfg.get("paths", {}).get("root_results", "results")
    exp_dir = os.path.join(root_results_dir, timestamp)
    logs_dir = os.path.join(exp_dir, "logs")
    _ensure_dir(exp_dir); _ensure_dir(logs_dir); _ensure_dir(os.path.join(exp_dir, "checkpoints"))

    runall_log = os.path.join(logs_dir, "run_all.log")
    _append(runall_log, "========== RUN_ALL START ==========")
    _append(runall_log, f"timestamp: {timestamp}")
    _append(runall_log, f"exp_dir:    {os.path.abspath(exp_dir)}")
    _append(runall_log, f"base_config:{os.path.abspath(args.config)}")

    cfg = dict(base_cfg)
    cfg.setdefault("paths", {})
    cfg["paths"]["save_dir"] = exp_dir
    cfg["paths"]["scaler"] = os.path.join(exp_dir, "scaler.pkl")

    exp_config_path = os.path.join(exp_dir, "config_used.yaml")
    with open(exp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    _ensure_dir("config")
    shutil.copy(exp_config_path, os.path.join("config", "current.yaml"))

    python_exec = sys.executable
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

    _append(runall_log, "\n===== STAGE: TRAIN =====")
    run_cmd([python_exec, "train.py", "--config", exp_config_path],
            stdout_path=train_out, stderr_path=train_err, runall_log=runall_log)

    _append(runall_log, "\n===== STAGE: PREDICT =====")
    run_cmd([python_exec, "export_results.py", "--config", exp_config_path, "--outdir", os.path.join(exp_dir, "exports")],
            stdout_path=pred_out, stderr_path=pred_err, runall_log=runall_log)

    shap_cfg = cfg.get("shap", {})
    shap_enabled = bool(shap_cfg.get("enabled", False))
    _append(runall_log, f"\n===== STAGE: SHAP (enabled={shap_enabled}, strict={args.shap_strict}) =====")

    shap_rc = None
    if shap_enabled:
        shap_rc = run_cmd([python_exec, "shap_analysis.py", "--config", exp_config_path],
                          stdout_path=shap_out, stderr_path=shap_err, runall_log=runall_log,
                          allow_fail=(not args.shap_strict))
        if shap_rc != 0:
            _append(runall_log, f"[run_all] SHAP failed (rc={shap_rc}). See:\n  {shap_err}\n  {os.path.join(exp_dir, 'shap', 'crash.log')}")
    else:
        _append(runall_log, "[run_all] SHAP disabled. Set shap.enabled: true in config to enable.")

    # ---- PT 可视化（HTML dashboard）----
    ptviz_cfg = cfg.get("pt_viz", {}) or {}
    ptviz_enabled = bool(ptviz_cfg.get("enabled", True))
    ptviz_allow_fail = bool(ptviz_cfg.get("allow_fail", True))
    ptviz_diff_mode = str(ptviz_cfg.get("diff_mode", "pred_minus_true"))

    _append(runall_log, f"\n===== STAGE: PT_VIZ (enabled={ptviz_enabled}, allow_fail={ptviz_allow_fail}, diff_mode={ptviz_diff_mode}) =====")

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

                    # --- Pred-vs-True Compare (Pretrain/Gate/Finetune in one HTML) ---
                    # Use the 6 clean exports as inputs; never writes extra CSV.
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
                # 单目标兼容：predict.py 会写 y_true/y_pred
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

                # single-target compare html
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

    summary_path = os.path.join(exp_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    summary.setdefault("RunAll", {})
    summary["RunAll"].update({
        "timestamp": timestamp,
        "exp_dir": os.path.abspath(exp_dir),
        "config_used": os.path.abspath(exp_config_path),
        "run_all_log": os.path.abspath(runall_log),
        "shap_enabled": shap_enabled,
        "shap_returncode": shap_rc,
        "logs": {
            "train_stdout": os.path.abspath(train_out),
            "train_stderr": os.path.abspath(train_err),
            "predict_stdout": os.path.abspath(pred_out),
            "predict_stderr": os.path.abspath(pred_err),
            "shap_stdout": os.path.abspath(shap_out),
            "shap_stderr": os.path.abspath(shap_err),
            "pt_viz_stdout": os.path.abspath(ptviz_out),
            "pt_viz_stderr": os.path.abspath(ptviz_err),
            "pt_predtrue_compare_stdout": os.path.abspath(ptcmp_out),
            "pt_predtrue_compare_stderr": os.path.abspath(ptcmp_err),
        },
    })

    
    # Keep only the 6 export CSVs; remove any other CSV artifacts created by submodules.
    exports_dir = os.path.join(exp_dir, "exports")
    keep_names = [
        "pretrain_best_predictions.csv",
        "pretrain_best_metrics.csv",
        "gate_test_predictions.csv",
        "gate_test_metrics.csv",
        "finetune_test_predictions.csv",
        "finetune_test_metrics.csv",
    ]
    keep_abs = set()
    for name in keep_names:
        keep_abs.add(os.path.abspath(os.path.join(exports_dir, name)))
    _cleanup_extra_csv(exp_dir, keep_abs_paths=keep_abs, runall_log=runall_log)

    summary.setdefault("PtVizArtifacts", {})
    summary["PtVizArtifacts"].update({
        "dashboards": ptviz_html_map,
        "pred_true_compare": ptcmp_html_map,
    })

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

        try:
            root = "results"
            candidate = "run_all_error.log"
            if os.path.isdir(root):
                subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
                if subdirs:
                    candidate = os.path.join(root, subdirs[-1], "logs", "run_all.log")
            _append(candidate, "\n❌ RUN_ALL FAILED")
            _append(candidate, tb)
        except Exception:
            pass
        sys.exit(1)