"""utils/distributed.py

Minimal, pragmatic DistributedDataParallel (DDP) helpers.

Why this exists:
  - If you launch this project with multiple processes (torchrun/Slurm) but do
    NOT initialize torch.distributed and do NOT use DistributedSampler,
    *each process will train on the full dataset*, producing identical logs and
    wasting compute.

This helper:
  - infers rank/world_size/local_rank from torchrun or Slurm env vars
  - initializes the process group (NCCL on CUDA)
  - sets the correct CUDA device for each local_rank
  - provides safe barrier()/destroy() utilities

Important for NCCL stability:
  - We pass device_id to init_process_group and device_ids to barrier() to avoid
    NCCL "guessing device" warnings and potential hangs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return (not self.enabled) or self.rank == 0


def _get_env_int(key: str, default: int | None) -> int | None:
    v = os.environ.get(key, None)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _infer_from_slurm() -> tuple[int, int, int]:
    """Infer (rank, world_size, local_rank) from Slurm env vars if present."""
    rank = int(_get_env_int("SLURM_PROCID", 0) or 0)
    world_size = int(_get_env_int("SLURM_NTASKS", 1) or 1)
    local_rank = int(_get_env_int("SLURM_LOCALID", 0) or 0)
    return rank, world_size, local_rank


def _infer_rank_info() -> tuple[int, int, int]:
    """Infer (rank, world_size, local_rank) from torchrun OR Slurm."""
    # torchrun sets these
    rank = _get_env_int("RANK", None)
    world_size = _get_env_int("WORLD_SIZE", None)
    local_rank = _get_env_int("LOCAL_RANK", None)

    if rank is None or world_size is None or local_rank is None:
        srank, sworld, slocal = _infer_from_slurm()
        rank = srank if rank is None else rank
        world_size = sworld if world_size is None else world_size
        local_rank = slocal if local_rank is None else local_rank

    rank = int(rank) if rank is not None else 0
    world_size = int(world_size) if world_size is not None else 1
    local_rank = int(local_rank) if local_rank is not None else 0
    return rank, world_size, local_rank


def init_distributed(backend: str | None = None, timeout_min: int = 30) -> DistInfo:
    """Initialize torch.distributed if launched with >1 processes.

    Returns DistInfo(enabled=False, ...) for normal single-process runs.
    """
    rank, world_size, local_rank = _infer_rank_info()
    enabled = world_size > 1
    if not enabled:
        return DistInfo(enabled=False, rank=0, world_size=1, local_rank=0)

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # env:// init needs MASTER_ADDR/MASTER_PORT.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")

    if not dist.is_initialized():
        # device_id avoids NCCL guessing device mapping (can hang on some setups).
        device_id = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=int(timeout_min)),
            device_id=device_id,
        )

    # Sync all ranks before continuing.
    barrier(local_rank=local_rank)
    return DistInfo(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def barrier(local_rank: int | None = None):
    """A safer barrier that pins the barrier to the correct CUDA device."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available():
        lr = int(local_rank) if local_rank is not None else int(_get_env_int("LOCAL_RANK", 0) or 0)
        dist.barrier(device_ids=[lr])
    else:
        dist.barrier()


def destroy_distributed(dist_info: DistInfo):
    if dist_info.enabled and dist.is_available() and dist.is_initialized():
        try:
            barrier(local_rank=dist_info.local_rank)
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
