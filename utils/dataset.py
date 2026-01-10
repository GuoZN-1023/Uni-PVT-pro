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


def infer_feature_and_target_cols(
    columns: list[str],
    anchor_col: str = "Z (-)",
    expert_col: str = "no",
) -> tuple[list[str], list[str]]:
    """
    智能识别列（按用户定义的“相对位置规则”）：

    - 自变量 X：在 anchor_col（默认 'Z (-)'）**之前** 的所有列（不含 anchor_col）
    - 因变量候选 Y：从 anchor_col（含）到 expert_col（默认 'no'）之前的所有列
      即 [anchor_col, ..., 直到 no 前一列]，不含 no

    该规则严格依赖 CSV 的**列顺序**（不是字母排序）。
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
    - 按“位置规则”推断特征列 X 与因变量候选列 Y
    - 通过 config 中的 targets/target_cols/target_col 选择要预测的目标列（可 1 个或多个）
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

        # 按列顺序推断 X / Y 候选
        feature_cols_raw, target_candidates_raw = infer_feature_and_target_cols(
            list(df_raw.columns), anchor_col=anchor_col, expert_col=expert_col
        )

        # 选择要预测的 targets（可多目标）
        targets_cfg = (
            cfg.get("targets", None)
            if "targets" in cfg
            else cfg.get("target_cols", None)
        )
        if targets_cfg is None and "target_col" in cfg and isinstance(cfg.get("target_col"), str):
            targets_cfg = cfg.get("target_col")
        chosen_targets = _normalize_targets_cfg(targets_cfg, target_candidates_raw)
        if len(chosen_targets) == 0:
            raise ValueError(
                "No valid targets selected. "
                f"Candidates (between {anchor_col} and {expert_col}): {target_candidates_raw}. "
                f"Your config targets: {targets_cfg}"
            )

        # 数值化：只保留需要的列（X + chosen_targets + expert_col）
        needed_cols = list(feature_cols_raw) + list(chosen_targets) + [expert_col]
        df = df_raw.loc[:, needed_cols].copy()

        # 强制转换为数值（把字符串等转成 NaN），再统一 dropna
        for c in needed_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        before = len(df)
        df = df.dropna(axis=0, how="any").reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            # 保持安静，但给出最基本的提示
            print(f"[ZDataset] Dropped {dropped} rows due to NaNs after numeric conversion.")

        # subset（可选）
        subset_cfg = cfg.get("subset", None)
        df = apply_subset(df, subset_cfg=subset_cfg, expert_col=expert_col).reset_index(drop=True)

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
    data_path = cfg["paths"]["data"]
    scaler_path = cfg["paths"]["scaler"]

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

    bs = int(cfg["training"]["batch_size"])
    return {
        "train": DataLoader(train_set, batch_size=bs, shuffle=True),
        "val": DataLoader(val_set, batch_size=bs, shuffle=False),
        "test": DataLoader(test_set, batch_size=bs, shuffle=False),
    }
