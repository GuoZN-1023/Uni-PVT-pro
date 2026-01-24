# Uni-PVT: A Neural Network for Universal Fluid Thermodynamic Property Prediction

Uni-PVT is a thermodynamic property prediction framework that targets the practical gap between engineering-speed cubic equations of state (EOS) and high-fidelity Helmholtz-energy EOS. The central objective is to learn a **universal neural equation of state** that remains accurate across phases, near criticality, and in phase-transition regimes, while preserving the runtime characteristics needed for large-scale engineering use.

The codebase implements one unified model and training pipeline, while allowing two complementary representations of molecular information. In a lightweight setting, the model consumes classical reduced-state descriptors and scalar molecular properties (for example *T_r, p_r, ω, T_c, p_c*). When richer structural information is needed, the same pipeline can instead consume cached 3D conformer embeddings produced offline by a SchNet encoder. In both cases, the downstream mixture-of-experts predictor, the physics-informed coupling, and the evaluation/export tooling remain consistent, which keeps ablations and comparisons clean and reproducible.

This README is written with the clarity and reproducibility norms of top-tier ML venues (ICLR/CVPR/NeurIPS style) in mind: it explains the method, identifies the key entrypoints, describes dataset and config conventions, and documents the experiment artifact layout used for reporting.

---

## Method overview

### Phase-aware Mixture-of-Experts backbone

Uni-PVT uses a phase-aware Mixture-of-Experts (MoE) predictor to handle regime-dependent thermodynamic behavior. A gate network produces per-sample routing weights \(w \in \mathbb{R}^{5}\). Each expert is an MLP that maps the input representation to the target vector (single-target or multi-target). The final prediction is a convex mixture of expert outputs:

\[
\hat{y}(x) = \sum_{k=1}^{5} w_k(x)\, f_k(x).
\]

The implementation fixes the number of experts to five, corresponding to a practical partition of regimes (commonly interpreted as gas, liquid, transition, supercritical, and critical). The region label column is read from `expert_col` (default: `no`) and is expected to be **1-based** (1..5). If the dataset uses a different convention, a normalization step should be applied during dataset preparation so that the model’s routing semantics are stable across experiments.

Implementation reference: `models/fusion_model.py` (MoE head, gate, and expert networks).

### Thermodynamic PINN coupling in the loss (not as extra input)

Uni-PVT augments supervised regression with physics-informed residual constraints computed by autograd. The coupling is enforced **in the objective**, rather than by repeatedly re-predicting intermediate quantities each iteration and appending them to the input as extra dimensions. When the necessary targets are present, the code supports residuals aligned with standard EOS identities, including relations between \(Z\), \(\ln\phi\), and pressure/temperature derivatives, and the pressure-derivative forms for enthalpy and entropy that depend on \(\partial Z/\partial T\).

Two differentiation modes are supported. In a feature-scaled mode, the loss accounts for feature standardization (typical in scalar-feature datasets with `StandardScaler`). In a state mode, the loss differentiates directly with respect to physical state variables \((T, P)\) exposed explicitly by the dataset (typical when the input representation is built from cached molecular embeddings and explicit state conditioning).

Implementation reference: `utils/thermo_pinn.py` (`ThermoPinnLoss`).

### Three-stage training strategy for stable MoE optimization

Training follows a deterministic three-stage schedule to improve stability and mitigate gate collapse. Experts are first pretrained under hard routing using the region label. The gate is then trained under soft routing (and experts may optionally be updated), with an optional temperature schedule for smoothing. Finally, the full model is finetuned jointly. The trainer writes stage-specific checkpoints and guarantees a canonical `best_model.pt` checkpoint so that evaluation and export scripts are not sensitive to the exact schedule choices.

Implementation reference: `utils/trainer.py` (`train_model`).

---

## Molecular representation: scalar descriptors or cached 3D conformer embeddings

Uni-PVT treats “molecular information” as a pluggable component.

In scalar mode, the configured `feature_cols` provide reduced-state descriptors and molecular scalars, which are then passed through the same backbone described above.

In embedding mode, each sample references a molecule by `mol_id`. An offline cache stores a conformer ensemble \((z_k, e_k)\) per molecule, where \(z_k\) is a SchNet embedding and \(e_k\) is a relative conformer energy. At runtime, conformers are pooled using Boltzmann weighting \(a_k \propto \exp(-\Delta e_k/(k_B T))\), yielding a single molecular vector. A Fourier-feature state embedding of \((T, P)\) is computed and used to FiLM-modulate the molecular vector before it is passed to the MoE predictor. The core training and evaluation pipeline remains unchanged, which makes it straightforward to compare scalar features versus 3D embeddings under identical losses, schedules, and metrics.

Implementation reference: `models/fusion_model.py` (pooling, Fourier embedding, FiLM fusion, and cache-driven feature construction).

### Offline cache builder

`build_mol_cache.py` reads a molecule table (CSV) containing `SMILES` and a molecule identifier (default `mol_id`). It generates \(K\) conformers per molecule (ETKDG), optionally optimizes them (MMFF/UFF when available), encodes each conformer with a SchNet backbone, and writes compact cache artifacts such as `z_conf.npy` (conformer embeddings), `e_conf.npy` (relative energies), and `meta.json` (dimensions, \(K\), and provenance). The script includes defensive fallbacks for molecules unsupported by specific force fields and accommodates SchNet / PyG API differences across versions.

---

## Repository structure

The repository is organized around a small set of entrypoints and a stable experiment artifact layout.

`train.py` is the main training entrypoint (with optional distributed support) and writes `config_used.yaml` into the experiment directory. `run_all.py` provides an end-to-end runner that can prepare datasets, optionally build molecular caches, train, evaluate, and export results in one command. `prepare_dataset.py` builds deterministic train/val/test splits from many per-molecule CSV files and can export compact formats. `predict.py` evaluates a trained model and writes both global metrics and region-wise analysis. `export_results.py` produces aligned prediction/ground-truth tables and summary artifacts under `exports/`. Model components live under `models/`, while datasets, training logic, and physics losses live under `utils/`.

---

## Environment and installation

The core stack is PyTorch-based. Scalar-feature training typically requires `torch`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, and `joblib`. The embedding workflow additionally requires `rdkit` and `torch_geometric` (PyG), ideally matched to your CUDA and PyTorch builds. Optional plotting utilities may rely on `plotly` for convenience.

A minimal installation typically follows:

```bash
pip install -r requirements.txt
```

For the embedding workflow, install RDKit and PyG according to your platform and CUDA configuration. A common Conda approach is:

```bash
conda install -c conda-forge rdkit
# Install torch-geometric following the official PyG instructions for your torch/cuda version.
```

---

## Data conventions and configuration semantics

Uni-PVT supports either fixed split files for maximal reproducibility or a single CSV with deterministic on-the-fly splitting.

If you provide explicit split files, set `paths.train_data`, `paths.val_data`, and `paths.test_data` to CSV (or the compact formats used by your dataset utilities). This is recommended because the split becomes an explicit artifact that can be versioned.

If you provide a single CSV, set `paths.data` to that file and configure split ratios under `data.split`, while controlling determinism via `training.seed`.

Column selection is intentionally flexible. Input features can be specified explicitly with `feature_cols`. Targets can be specified explicitly via `targets: [ ... ]`. When `targets` is omitted, the loader can fall back to a legacy anchor rule that infers targets from a contiguous column block (from `anchor_col` inclusive, default `Z (-)`, up to `expert_col` exclusive, default `no`). Region labels are read from `expert_col` (default `no`) and are expected to be 1..5.

Implementation reference: `utils/dataset.py`, `prepare_dataset.py`.

---

## Quickstart

### End-to-end pipeline (recommended)

`run_all.py` creates a timestamped experiment directory under your configured results root, then writes logs, configs, checkpoints, evaluation outputs, and exports under a consistent layout:

```bash
python run_all.py --config configs/config_total_Z.yaml
```

Depending on the config, the runner can prepare split datasets, build molecular caches, train with three-stage scheduling, evaluate, and export paper-ready tables.

### Training only

If you already have split files and only need training:

```bash
python train.py --config configs/config_total_Z.yaml
```

A successful run writes logs, a resolved `config_used.yaml`, and `checkpoints/best_model.pt` under `paths.save_dir`.

### Evaluation

```bash
python predict.py --config <path_to_experiment>/config_used.yaml
```

Evaluation outputs are written under `<save_dir>/eval/` and `<save_dir>/region_eval/`.

---

## Using cached molecular embeddings

If your dataset rows reference molecules by `mol_id` and expose explicit state columns (commonly `T (K)` and `P (Pa)`), you can build a cache and point the training config to it.

Build the cache:

```bash
python build_mol_cache.py   --csv <mol_table.csv>   --outdir <mol_cache_dir>   --smiles_col SMILES   --mol_id_col mol_id   --k 10   --embed_dim 128   --device cuda
```

Then configure the cache in your YAML (names may vary slightly across configs):

```yaml
mol_encoder:
  enabled: true
  mol_id_col: mol_id
  state_cols: ["T (K)", "P (Pa)"]
  cache:
    z_conf_path: /path/to/mol_cache/z_conf.npy
    e_conf_path: /path/to/mol_cache/e_conf.npy
    meta_path:   /path/to/mol_cache/meta.json
```

Training and evaluation proceed exactly as in scalar mode. When the dataset exposes physical \((T,P)\) directly, the PINN loss can operate in state-differentiation mode (unless overridden).

---

## Checkpoints and artifacts

Each experiment directory is designed to be self-sufficient for reproduction and reporting. Logs live under `logs/`. Checkpoints live under `checkpoints/` and include stage-specific best models and a canonical `best_model.pt`. Exported tables and summary artifacts are written under `exports/` via `export_results.py` or by the end-to-end runner.

A resolved `config_used.yaml` is always emitted and should be treated as the authoritative record of a run, since it includes inferred dimensions and the final resolved feature and target columns.

---

## Citation

If you use Uni-PVT in academic work, please cite the accompanying paper (BibTeX placeholder):

```bibtex
@inproceedings{unipvt2026,
  title     = {Uni-PVT: A Universal Neural Equation of State with Phase-Aware Mixture-of-Experts and Thermodynamic PINN Coupling},
  author    = {Author(s) to be filled},
  booktitle = {Proceedings of ...},
  year      = {2026}
}
```
