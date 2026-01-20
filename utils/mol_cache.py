"""utils/mol_cache.py

Fast, reliable *offline* molecular embedding cache I/O.

Why this exists
--------------
Uni-PVT 2.0 wants a heavy 3D encoder (e.g. SchNet on conformer ensembles), but we do NOT
want to run it inside every training step. The practical workflow is:

  1) Build a cache once: (mol_id -> {z_conf[K,D], e_conf[K]}).
  2) During training, the Dataset simply reads cached arrays and returns them.
  3) The model performs only lightweight Boltzmann pooling + FiLM + MoE.

This file implements a cache reader that supports memory-mapped .npy arrays.

Cache format (recommended)
--------------------------
We store two arrays:

  - z_conf.npy  : float16/float32, shape (N, K, D)
  - e_conf.npy  : float16/float32, shape (N, K)

Optionally a JSON meta file:

  {
    "num_mols": N,
    "k_conformers": K,
    "embed_dim": D,
    "dtype": "float16",
    "id_map": "mol_id_to_row.json"   # optional
  }

If id_map is provided, it must map str(mol_id) -> row_index (int).
If absent, mol_id is assumed to be the row index.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class MolCacheMeta:
    num_mols: int
    k_conformers: int
    embed_dim: int
    dtype: str
    z_conf_path: str
    e_conf_path: str
    id_map_path: Optional[str] = None


class MolCache:
    """Memory-mapped cache reader.

    Notes on multiprocessing:
      - We keep only file paths in the pickled state.
      - Each worker process lazily opens its own np.memmap views.
    """

    def __init__(
        self,
        *,
        z_conf_path: str,
        e_conf_path: str,
        meta_path: str | None = None,
        id_map_path: str | None = None,
    ):
        if not os.path.exists(z_conf_path):
            raise FileNotFoundError(f"MolCache z_conf not found: {z_conf_path}")
        if not os.path.exists(e_conf_path):
            raise FileNotFoundError(f"MolCache e_conf not found: {e_conf_path}")

        self.z_conf_path = str(z_conf_path)
        self.e_conf_path = str(e_conf_path)
        self.meta_path = str(meta_path) if meta_path else None

        meta = None
        if self.meta_path and os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # Resolve id_map from meta unless explicitly given.
        if id_map_path is None and isinstance(meta, dict):
            id_map_path = meta.get("id_map", None) or meta.get("index_path", None)
            if id_map_path and (not os.path.isabs(id_map_path)) and self.meta_path:
                id_map_path = os.path.join(os.path.dirname(self.meta_path), id_map_path)
        self.id_map_path = str(id_map_path) if id_map_path else None

        # Lazy handles
        self._z_mm: Optional[np.memmap] = None
        self._e_mm: Optional[np.memmap] = None
        self._id_map: Optional[Dict[str, int]] = None

        # Infer shape/dtype from npy header without loading the full array.
        z0 = np.load(self.z_conf_path, mmap_mode="r")
        e0 = np.load(self.e_conf_path, mmap_mode="r")

        if z0.ndim != 3:
            raise ValueError(f"Expected z_conf shape (N,K,D), got {z0.shape}")
        if e0.ndim != 2:
            raise ValueError(f"Expected e_conf shape (N,K), got {e0.shape}")
        if z0.shape[0] != e0.shape[0] or z0.shape[1] != e0.shape[1]:
            raise ValueError(
                f"z_conf and e_conf shape mismatch: z={z0.shape}, e={e0.shape}"
            )

        self.num_mols = int(z0.shape[0])
        self.k_conformers = int(z0.shape[1])
        self.embed_dim = int(z0.shape[2])
        self.dtype = str(z0.dtype)

    def __getstate__(self):
        # Drop live memmaps / maps when pickling to workers.
        d = dict(self.__dict__)
        d["_z_mm"] = None
        d["_e_mm"] = None
        d["_id_map"] = None
        return d

    def _ensure_open(self):
        if self._z_mm is None:
            self._z_mm = np.load(self.z_conf_path, mmap_mode="r")
        if self._e_mm is None:
            self._e_mm = np.load(self.e_conf_path, mmap_mode="r")

        if self._id_map is None and self.id_map_path:
            if not os.path.exists(self.id_map_path):
                raise FileNotFoundError(f"MolCache id_map not found: {self.id_map_path}")
            # Support either JSON (mol_id -> row) or CSV (mol_id,row).
            if self.id_map_path.lower().endswith(".csv"):
                m: Dict[str, int] = {}
                with open(self.id_map_path, "r", encoding="utf-8") as f:
                    header = f.readline()
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 2:
                            continue
                        mol_id, row = parts[0], parts[1]
                        if mol_id == "" or row == "":
                            continue
                        m[str(mol_id)] = int(row)
                self._id_map = m
            else:
                with open(self.id_map_path, "r", encoding="utf-8") as f:
                    self._id_map = json.load(f)

    def _row_index(self, mol_id: int | str) -> int:
        self._ensure_open()
        if self._id_map is None:
            try:
                rid = int(mol_id)
            except Exception as e:
                raise ValueError(
                    "MolCache: id_map is not provided, so mol_id must be an int row index. "
                    f"Got mol_id={mol_id!r}"
                ) from e
            if rid < 0 or rid >= self.num_mols:
                raise IndexError(f"MolCache: mol_id(row) out of range: {rid} not in [0,{self.num_mols})")
            return rid

        key = str(mol_id)
        if key not in self._id_map:
            raise KeyError(f"MolCache: mol_id {mol_id!r} not found in id_map")
        rid = int(self._id_map[key])
        if rid < 0 or rid >= self.num_mols:
            raise IndexError(f"MolCache: mapped row out of range: {rid} not in [0,{self.num_mols})")
        return rid

    def get(self, mol_id: int | str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (z_conf[K,D], e_conf[K]) as *numpy* views (memmap slices)."""
        rid = self._row_index(mol_id)
        assert self._z_mm is not None and self._e_mm is not None
        z = self._z_mm[rid]  # (K,D)
        e = self._e_mm[rid]  # (K,)
        return z, e

    def get_meta(self) -> MolCacheMeta:
        return MolCacheMeta(
            num_mols=self.num_mols,
            k_conformers=self.k_conformers,
            embed_dim=self.embed_dim,
            dtype=self.dtype,
            z_conf_path=self.z_conf_path,
            e_conf_path=self.e_conf_path,
            id_map_path=self.id_map_path,
        )
