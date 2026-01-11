# utils/dataset.py
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_subset(df: pd.DataFrame, subset_cfg: dict | None,
                 expert_col: str | None) -> pd.DataFrame:
    """
    根据 config['subset'] 在 DataFrame 层面筛选数据：
      - include_expert_ids / exclude_expert_ids
      - fraction / max_samples

    注意：subset 是“可选的训练加速/调试工具”，不影响列推断逻辑。
    """
    if not subset_cfg or not subset_cfg.get("enabled", False):
        return df

    mask = pd.Series(True, index=df.index)

    # 按 no 等编号筛选
    if expert_col and expert_col in df.columns:
        inc = subset_cfg.get("include_expert_ids", None)
        exc = subset_cfg.get("exclude_expert_ids", None)
        if inc is not None:
            inc = set(int(x) for x in inc)
            mask &= df[expert_col].astype(int).isin(inc)
        if exc is not None:
            exc = set(int(x) for x in exc)
            mask &= ~df[expert_col].astype(int).isin(exc)

    df = df.loc[mask].copy()

    # 随机抽样 fraction
    frac = subset_cfg.get("fraction", None)
    if frac is not None:
        frac = float(frac)
        frac = max(min(frac, 1.0), 0.0)
        if 0.0 < frac < 1.0:
            df = df.sample(frac=frac, random_state=42)

    # 最多 max_samples
    max_samples = subset_cfg.get("max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

    return df


def _get_cfg(cfg: dict, key: str, default=None):
    """Allow both flat config and nested `data:` config."""
    if cfg is None:
        return default
    if key in cfg:
        return cfg.get(key, default)
    data = cfg.get("data", None)
    if isinstance(data, dict) and key in data:
        return data.get(key, default)
    return default


def infer_feature_and_target_cols(
    columns: list[str],
    anchor_col: str = "Z (-)",
    expert_col: str = "no",
) -> tuple[list[str], list[str]]:
    """
    兼容旧逻辑：基于列顺序的“锚定推断”。

    - 自变量 X：在 anchor_col 前的所有列（不含 anchor_col）
    - 因变量候选 Y：从 anchor_col（含）到 expert_col 前（不含 expert_col）的所有列

    注意：严格依赖 CSV 的列顺序。
    """
    cols = list(columns)
    if anchor_col not in cols:
        raise ValueError(f"Anchor column '{anchor_col}' not found in CSV columns.")
    if expert_col not in cols:
        raise ValueError(f"Expert/region column '{expert_col}' not found in CSV columns.")

    i_anchor = cols.index(anchor_col)
    i_expert = cols.index(expert_col)
    if i_expert <= i_anchor:
        raise ValueError(
            f"Column order invalid: '{expert_col}' must appear AFTER '{anchor_col}'. "
            f"(got idx({expert_col})={i_expert}, idx({anchor_col})={i_anchor})"
        )

    feature_cols = cols[:i_anchor]
    target_candidate_cols = cols[i_anchor:i_expert]  # includes anchor, excludes 'no'
    if len(feature_cols) == 0:
        raise ValueError("No feature columns inferred (nothing appears before anchor_col).")
    if len(target_candidate_cols) == 0:
        raise ValueError("No target columns inferred (nothing between anchor_col and expert_col).")
    return feature_cols, target_candidate_cols


def resolve_columns_from_config(
    columns: list[str],
    cfg: dict,
) -> tuple[list[str], list[str], str]:
    """
    新逻辑：显式指定 feature_cols / targets（列名），支持任意输入维度与任意多目标。

    规则：
    - expert_col（默认 'no'）必须存在，用作相区编号；不会被加入 X / Y。
    - 若 cfg 中显式提供 feature_cols，则按该列表作为 X。
    - 若 cfg 中显式提供 targets，则按该列表作为 Y（可多列）。
    - 若未显式提供 feature_cols 或 targets，则回退到“锚定推断”(anchor_col) 的旧逻辑。
    """
    cols = list(columns)

    expert_col = _get_cfg(cfg, "expert_col", "no")
    if expert_col not in cols:
        raise ValueError(f"Required expert/region column '{expert_col}' not found in CSV columns.")

    # 读取显式配置
    feature_cols_cfg = _get_cfg(cfg, "feature_cols", None)
    if feature_cols_cfg is None:
        feature_cols_cfg = _get_cfg(cfg, "features", None)

    targets_cfg = _get_cfg(cfg, "targets", None)
    if targets_cfg is None:
        targets_cfg = _get_cfg(cfg, "target_cols", None)
    if targets_cfg is None:
        targets_cfg = _get_cfg(cfg, "target_col", None)

    # 标准化配置形态
    def _norm_list(v):
        if v is None:
            return None
        if isinstance(v, str):
            v_strip = v.strip()
            # allow "a,b,c"
            if "," in v_strip and v_strip.lower() not in ("all", "*"):
                return [s.strip() for s in v_strip.split(",") if s.strip()]
            return v_strip
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return v

    feature_cols_cfg = _norm_list(feature_cols_cfg)
    targets_cfg = _norm_list(targets_cfg)

    # 如果显式给了 features/targets，就用显式；否则回退 anchor 逻辑
    anchor_col = _get_cfg(cfg, "anchor_col", "Z (-)")
    use_anchor_fallback = False
    if feature_cols_cfg is None or targets_cfg is None:
        # 只有在能够 anchor 推断时才回退，否则要求用户显式给出
        if (anchor_col in cols) and (expert_col in cols) and (cols.index(expert_col) > cols.index(anchor_col)):
            use_anchor_fallback = True
        else:
            missing = []
            if feature_cols_cfg is None:
                missing.append("feature_cols")
            if targets_cfg is None:
                missing.append("targets")
            raise ValueError(
                "Column selection requires explicit config when anchor-based inference is unavailable. "
                f"Missing: {missing}. Please set them in config."
            )

    if use_anchor_fallback:
        f_raw, t_candidates = infer_feature_and_target_cols(cols, anchor_col=anchor_col, expert_col=expert_col)
        # feature_cols: 如果用户显式指定了则覆盖，否则使用推断
        feature_cols = f_raw if feature_cols_cfg is None else (
            t_candidates if isinstance(feature_cols_cfg, str) and feature_cols_cfg.lower() in ("all", "*") else feature_cols_cfg
        )
        # targets: 如果用户没给，则默认 all candidates；若给了 "all" 则 all candidates；否则按列表挑选
        all_targets = list(t_candidates)
        targets = _normalize_targets_cfg(targets_cfg, all_targets)
    else:
        # 显式模式
        # 显式模式：允许 targets 或 feature_cols 使用 'all' 作为快捷方式
        # - targets='all': 预测除 feature_cols 与 expert_col 之外的全部列
        # - feature_cols='all': 输入除 targets 与 expert_col 之外的全部列
        # 但不允许二者同时为 'all'（会变成循环定义）。
        if isinstance(feature_cols_cfg, str) and feature_cols_cfg.lower() in ("all", "*") and isinstance(targets_cfg, str) and targets_cfg.lower() in ("all", "*"):
            raise ValueError("feature_cols and targets cannot both be 'all' in explicit mode.")
        if isinstance(feature_cols_cfg, str):
            if feature_cols_cfg.lower() in ("all", "*"):
                feature_cols = None  # resolve later
            else:
                feature_cols = [feature_cols_cfg]
        else:
            feature_cols = list(feature_cols_cfg) if feature_cols_cfg is not None else []

        if isinstance(targets_cfg, str):
            if targets_cfg.lower() in ("all", "*"):
                targets = None  # resolve later
            else:
                targets = [targets_cfg]
        else:
            targets = list(targets_cfg) if targets_cfg is not None else []

        # resolve shortcuts
        if feature_cols is None and targets is None:
            raise ValueError("feature_cols and targets cannot both be 'all' in explicit mode.")

        if targets is None:
            # all columns except features & expert_col
            targets = [c for c in cols if c != expert_col and (feature_cols is None or c not in feature_cols)]
        if feature_cols is None:
            # all columns except targets & expert_col
            feature_cols = [c for c in cols if c != expert_col and c not in targets]

    # 校验：存在、无重复、并排除 expert_col
    def _check_list(name, lst):
        if not lst:
            raise ValueError(f"{name} resolved to empty. Please check your config.")
        seen = set()
        for c in lst:
            if c == expert_col:
                raise ValueError(f"Column '{expert_col}' is expert_col and cannot be used as {name}.")
            if c not in cols:
                raise ValueError(f"Column '{c}' listed in {name} not found in CSV columns.")
            if c in seen:
                raise ValueError(f"Duplicate column '{c}' in {name}.")
            seen.add(c)

    _check_list("feature_cols", feature_cols)
    _check_list("targets", targets)

    # 防止 X 与 Y 重叠
    overlap = set(feature_cols).intersection(set(targets))
    if overlap:
        raise ValueError(f"feature_cols and targets overlap: {sorted(overlap)}")

    return feature_cols, targets, expert_col

def _normalize_targets_cfg(targets_cfg, all_targets: list[str]) -> list[str]:
    """
    targets_cfg 支持：
      - None / "all" / "*" -> 全部因变量候选
      - str -> 单个
      - list[str] -> 多个
    """
    if targets_cfg is None:
        return list(all_targets)

    if isinstance(targets_cfg, str):
        if targets_cfg.strip().lower() in ("all", "*"):
            return list(all_targets)
        return [targets_cfg]

    if isinstance(targets_cfg, (list, tuple)):
        # 保留在 all_targets 中出现的、同时保持用户给出的顺序
        out = []
        for t in targets_cfg:
            if t in all_targets and t not in out:
                out.append(t)
        return out

    raise ValueError(f"Unsupported targets config type: {type(targets_cfg)}")


class ZDataset(Dataset):
    """
    自适应多任务数据集：

    - 读取 CSV
    - 优先按 config 指定的 feature_cols / targets（列名）构建 X / y（支持任意输入维度与多目标）
    - 若未指定 feature_cols 或 targets，可回退到旧的 anchor_col 位置规则推断（用于兼容）
    - no 列必须存在：作为相区编号 expert_id，仅用于 Stage1 专家预训练/区域加权，不进入特征

    __getitem__ 返回 (X, y, expert_id)
      - X: FloatTensor, shape (D,)
      - y: FloatTensor, shape (T,)  (T=目标数)
      - expert_id: LongTensor, shape ()
    """
    def __init__(
        self,
        csv_path: str,
        scaler_path: str,
        cfg: dict,
        train: bool = True,
    ):
        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.cfg = cfg
        self.csv_path = csv_path
        self.scaler_path = scaler_path
        self.train = bool(train)

        # 兼容旧配置：target_col 单列
        anchor_col = cfg.get("anchor_col", cfg.get("target_anchor_col", cfg.get("target_col", "Z (-)")))
        expert_col = cfg.get("expert_col", "no")

        df_raw = pd.read_csv(csv_path)

        # 解析列选择：优先使用 config 中的 feature_cols / targets（列名），否则回退锚定推断
        feature_cols_raw, chosen_targets, expert_col = resolve_columns_from_config(
            list(df_raw.columns), cfg=cfg
        )

        # 为了报告信息：尽量保留“候选池”；若无法推断则等于 chosen_targets
        try:
            _, target_candidates_raw = infer_feature_and_target_cols(
                list(df_raw.columns),
                anchor_col=_get_cfg(cfg, "anchor_col", "Z (-)"),
                expert_col=expert_col,
            )
        except Exception:
            target_candidates_raw = list(chosen_targets)

        # ---- 需要的列（X + targets + expert_col）----
        needed_cols = list(feature_cols_raw) + list(chosen_targets) + [expert_col]

        # 1) 训练/预测所用 df：仅包含 needed_cols，并且确保这些列都是可数值化的
        df_needed = df_raw.loc[:, needed_cols].copy()
        for c in needed_cols:
            df_needed[c] = pd.to_numeric(df_needed[c], errors="coerce")

        before = len(df_needed)
        df_needed = df_needed.dropna(axis=0, how="any")
        dropped = before - len(df_needed)
        if dropped > 0:
            # 保持安静，但给出最基本的提示
            print(f"[ZDataset] Dropped {dropped} rows due to NaNs after numeric conversion.")

        # subset（可选）
        subset_cfg = cfg.get("subset", None)
        df_needed = apply_subset(df_needed, subset_cfg=subset_cfg, expert_col=expert_col)

        # 2) 输出用 num_df：尽量保持旧项目行为（把“状态变量”一并带进 test_predictions.csv）
        # - 取 CSV 中原本就为数值的列
        # - 再把 needed_cols 强制数值化写回，确保这些列一定包含在输出里
        num_df = df_raw.select_dtypes(include=[np.number]).copy()
        for c in needed_cols:
            num_df[c] = pd.to_numeric(df_raw[c], errors="coerce")

        keep_idx = df_needed.index
        self.num_df = num_df.loc[keep_idx].reset_index(drop=True)

        # 最终用于建模的数据 df（只含 needed_cols），对齐 num_df 的行顺序
        df = df_needed.reset_index(drop=True)

        # expert id
        if expert_col not in df.columns:
            raise ValueError(f"Required expert_col '{expert_col}' missing after preprocessing.")
        self.expert_ids = df[expert_col].astype(int).values

        # 保存列信息（用于 train/predict 输出）
        self.anchor_col = anchor_col
        self.expert_col = expert_col
        self.feature_cols = list(feature_cols_raw)
        self.all_target_cols = list(target_candidates_raw)
        self.target_cols = list(chosen_targets)

        # 输出用 DataFrame（用于 predict.py 导出）
        # - self.num_df: 尽量保留 CSV 中所有可数值化列（与你旧版更一致，便于回溯与分析）
        # - self.feature_df: 仅包含 feature_cols（方便 SHAP / 只看输入特征时使用）
        self.feature_df = df[self.feature_cols].copy()

        # X / y
        X = df[self.feature_cols].values.astype(np.float32)
        y = df[self.target_cols].values.astype(np.float32)  # (N,T)

        # scaler
        if self.train:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            joblib.dump(scaler, self.scaler_path)
        else:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler not found: {self.scaler_path} (run train first?)")
            scaler = joblib.load(self.scaler_path)
            Xs = scaler.transform(X)


        # store scaler params for PINN derivative unit conversion
        self.scaler_mean = np.asarray(getattr(scaler, 'mean_', np.zeros(X.shape[1])), dtype=np.float32)
        self.scaler_scale = np.asarray(getattr(scaler, 'scale_', np.ones(X.shape[1])), dtype=np.float32)
        self.X = Xs.astype(np.float32)
        self.y = y
        self.input_dim = int(self.X.shape[1])
        self.output_dim = int(self.y.shape[1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        expert_id = torch.tensor(self.expert_ids[idx], dtype=torch.long)
        return x, y, expert_id


def get_dataloaders(cfg: dict):
    """
    根据 config 构建 train/val/test DataLoader

    重要：input_dim/output_dim 由数据自动推断，
    train.py 会在拿到 dataloaders 后把它写回 cfg['model']。
    """
    paths = cfg.get("paths", {}) or {}
    data_path = paths.get("data", None)
    scaler_path = paths.get("scaler", None)
    if not data_path:
        raise ValueError("Missing paths.data in config. Please set paths.data to your CSV file path.")
    if not scaler_path:
        # default to save_dir/scaler.pkl if not provided
        save_dir = paths.get("save_dir", "results/latest")
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        paths["scaler"] = scaler_path
        cfg["paths"] = paths

    set_seed(42)

    full_dataset = ZDataset(
        csv_path=data_path,
        scaler_path=scaler_path,
        cfg=cfg,
        train=True,
    )

    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=g
    )

    # -----------------------------
    # Multi-target scale statistics
    # -----------------------------
    # Multi-target regression often mixes variables with very different units.
    # Instead of forcing the network to chase the largest-magnitude target,
    # we compute per-target scales (std on the TRAIN split) and store them in cfg.
    # The loss can then use (pred-target)/scale for a balanced objective, while
    # model outputs remain in physical units (PINN stays consistent).
    try:
        # `random_split` returns Subset with .indices
        train_indices = getattr(train_set, "indices", None)
        if train_indices is None:
            train_indices = list(range(len(train_set)))

        y_all = getattr(full_dataset, "y", None)
        tcols = list(getattr(full_dataset, "target_cols", []))
        if y_all is not None and tcols:
            y_tr = y_all[np.asarray(train_indices, dtype=np.int64)]
            # std per target
            std = np.nanstd(y_tr, axis=0).astype(np.float32)
            std = np.where(std < 1e-12, 1.0, std)

            cfg.setdefault("loss", {})
            cfg["loss"].setdefault("target_scale_mode", "std")
            cfg["loss"]["target_scales"] = {tcols[i]: float(std[i]) for i in range(len(tcols))}
    except Exception as e:
        # Keep dataloader creation robust; if this fails, training can still proceed.
        print(f"[get_dataloaders] Warning: failed to compute target scales: {e}")

    bs = int((cfg.get("training", {}) or {}).get("batch_size", 256))
    return {
        "train": DataLoader(train_set, batch_size=bs, shuffle=True),
        "val": DataLoader(val_set, batch_size=bs, shuffle=False),
        "test": DataLoader(test_set, batch_size=bs, shuffle=False),
    }
