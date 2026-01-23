"""build_mol_cache.py

Offline builder for Uni-PVT 2.0 cached molecular encodings.

What it builds
--------------
For each molecule (row in the CSV), we produce:

  - z_conf: (K, D) conformer embeddings (SchNet graph-level outputs)
  - e_conf: (K,) conformer relative energies

and write two NumPy arrays (memmap friendly) + a small JSON meta file.

Why not CSV
-----------
CSV is the wrong container for dense float tensors: it's huge, slow to parse, and loses
dtype control. We store embeddings as .npy so you can memory-map them and stream fast.

Usage
-----
python build_mol_cache.py --csv data.csv --outdir mol_cache --smiles_col SMILES --mol_id_col mol_id

Dependencies
------------
rdkit, numpy, torch, torch_geometric
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import torch


def _try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except Exception as e:
        raise RuntimeError(
            "RDKit is required for conformer generation. Install via conda: conda install -c conda-forge rdkit"
        ) from e


def _try_import_schnet():
    try:
        from torch_geometric.nn.models import SchNet
        return SchNet
    except Exception as e:
        raise RuntimeError(
            "torch_geometric is required for SchNet encoding. Please install PyG matching your torch build."
        ) from e



def _build_schnet_encoder(SchNet, cfg: "BuildConfig"):
    """Create a SchNet that outputs a graph-level embedding of size cfg.embed_dim.

    PyG SchNet API differs across versions. Newer versions accept `out_channels`;
    older ones don't (they default to scalar output). We detect the signature and
    patch the final linear layer when needed so the output becomes (B, embed_dim).
    """
    import inspect
    import torch.nn as nn

    sig = inspect.signature(SchNet.__init__)
    params = set(sig.parameters.keys())

    base_kwargs = {}
    # only pass args that exist in this SchNet version
    if "hidden_channels" in params:
        base_kwargs["hidden_channels"] = int(cfg.embed_dim)
    if "num_filters" in params:
        base_kwargs["num_filters"] = int(cfg.embed_dim)
    if "num_interactions" in params:
        base_kwargs["num_interactions"] = int(cfg.num_interactions)
    if "num_gaussians" in params:
        base_kwargs["num_gaussians"] = int(cfg.num_gaussians)
    if "cutoff" in params:
        base_kwargs["cutoff"] = float(cfg.cutoff)

    if "out_channels" in params:
        return SchNet(out_channels=int(cfg.embed_dim), **base_kwargs)

    model = SchNet(**base_kwargs)

    # Patch the last head layer to output embed_dim
    if hasattr(model, "lin2") and isinstance(model.lin2, nn.Linear):
        in_f = int(model.lin2.in_features)
        model.lin2 = nn.Linear(in_f, int(cfg.embed_dim), bias=True)
    elif hasattr(model, "lin") and isinstance(model.lin, nn.Linear):
        in_f = int(model.lin.in_features)
        model.lin = nn.Linear(in_f, int(cfg.embed_dim), bias=True)
    else:
        raise RuntimeError(
            "Unsupported SchNet variant: cannot locate final head (lin2/lin) to patch for embedding output."
        )
    return model

@dataclass
class BuildConfig:
    k_conformers: int = 10
    embed_dim: int = 128
    cutoff: float = 10.0
    num_interactions: int = 6
    num_gaussians: int = 50
    energy_unit: str = "kcal/mol"  # RDKit MMFF energies are typically in kcal/mol


def _mol_from_smiles(Chem, smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return mol


def _generate_conformers(AllChem, mol, k: int, seed: int = 0) -> List[int]:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    # pruneRmsThresh helps create diverse conformers; keep it modest to avoid dropping too many.
    params.pruneRmsThresh = 0.2
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(k), params=params))
    return conf_ids


def _mmff_optimize(AllChem, mol, conf_id: int) -> float:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if props is None:
        # fallback to UFF if MMFF not available
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        return float(ff.CalcEnergy())
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", confId=conf_id, maxIters=500)
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    return float(ff.CalcEnergy())


def _extract_z_pos(Chem, mol, conf_id: int) -> Tuple[np.ndarray, np.ndarray]:
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    z = np.empty((n,), dtype=np.int64)
    pos = np.empty((n, 3), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        z[i] = int(atom.GetAtomicNum())
        p = conf.GetAtomPosition(i)
        pos[i, 0] = float(p.x)
        pos[i, 1] = float(p.y)
        pos[i, 2] = float(p.z)
    return z, pos


@torch.no_grad()
def _encode_conformer(schnet, z: np.ndarray, pos: np.ndarray) -> np.ndarray:
    # SchNet expects atomic numbers as torch.long and positions as float
    z_t = torch.from_numpy(z).long().to(next(schnet.parameters()).device)
    pos_t = torch.from_numpy(pos).float().to(next(schnet.parameters()).device)
    # batch=None -> single molecule
    emb = schnet(z_t, pos_t)
    # emb shape: [1, D]
    return emb.detach().cpu().numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV containing SMILES and mol_id")
    ap.add_argument("--outdir", required=True, help="Output directory for cache files")
    ap.add_argument("--smiles_col", default="SMILES", help="Column name for SMILES")
    ap.add_argument("--mol_id_col", default="mol_id", help="Column name for molecule id")
    ap.add_argument("--k", type=int, default=10, help="Number of conformers per molecule")
    ap.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension (SchNet out_channels)")
    ap.add_argument("--device", default=None, help="cpu/cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    Chem, AllChem = _try_import_rdkit()
    SchNet = _try_import_schnet()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"CSV missing smiles_col={args.smiles_col}")
    if args.mol_id_col not in df.columns:
        raise ValueError(f"CSV missing mol_id_col={args.mol_id_col}")

    mol_ids = df[args.mol_id_col].astype(str).tolist()
    smiles_list = df[args.smiles_col].astype(str).tolist()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg = BuildConfig(k_conformers=int(args.k), embed_dim=int(args.embed_dim))
    schnet = _build_schnet_encoder(SchNet, cfg).to(device)
    schnet.eval()

    n_mols = len(mol_ids)
    K, D = cfg.k_conformers, cfg.embed_dim

    z_conf = np.zeros((n_mols, K, D), dtype=np.float16)
    e_conf = np.zeros((n_mols, K), dtype=np.float32)

    failures = 0
    for i, (mid, smi) in enumerate(zip(mol_ids, smiles_list)):
        mol = _mol_from_smiles(Chem, smi)
        if mol is None:
            failures += 1
            continue
        conf_ids = _generate_conformers(AllChem, mol, K, seed=args.seed + i)
        if len(conf_ids) == 0:
            failures += 1
            continue

        # if fewer than K conformers, repeat last
        while len(conf_ids) < K:
            conf_ids.append(conf_ids[-1])

        energies = []
        embs = []
        for j, cid in enumerate(conf_ids[:K]):
            try:
                E = _mmff_optimize(AllChem, mol, cid)
                z, pos = _extract_z_pos(Chem, mol, cid)
                emb = _encode_conformer(schnet, z, pos)
            except Exception:
                E = float("nan")
                emb = np.zeros((D,), dtype=np.float32)
            energies.append(E)
            embs.append(emb)

        energies = np.asarray(energies, dtype=np.float32)
        # relative energies: shift min to 0 (more stable numerically)
        if np.isfinite(energies).any():
            energies = energies - np.nanmin(energies)
            energies = np.nan_to_num(energies, nan=float(np.nanmax(energies) if np.isfinite(energies).any() else 0.0))
        else:
            energies[:] = 0.0

        e_conf[i, :] = energies
        z_conf[i, :, :] = np.asarray(embs, dtype=np.float32).astype(np.float16)

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{n_mols}] built cache")

    # Write arrays
    z_path = os.path.join(args.outdir, "z_conf.npy")
    e_path = os.path.join(args.outdir, "e_conf.npy")
    np.save(z_path, z_conf)
    np.save(e_path, e_conf)

    # Write index map (mol_id -> row)
    index_path = os.path.join(args.outdir, "mol_index.csv")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("mol_id,row\n")
        for row, mid in enumerate(mol_ids):
            f.write(f"{mid},{row}\n")

    # Meta
    meta = {
        "k_conformers": K,
        "embed_dim": D,
        "dtype": "float16",
        "energy_unit": cfg.energy_unit,
        "energy_unit_scale_to_J_per_mol": 4184.0,  # kcal/mol -> J/mol
        "z_conf_path": "z_conf.npy",
        "e_conf_path": "e_conf.npy",
        "index_path": "mol_index.csv",
        "failures": failures,
    }
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Cache written to: {args.outdir}")
    print(f"  z_conf: {z_path}")
    print(f"  e_conf: {e_path}")
    print(f"  meta:   {os.path.join(args.outdir, 'meta.json')}")


if __name__ == "__main__":
    main()
