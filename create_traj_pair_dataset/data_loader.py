import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


class LoadedDataset:
    def __init__(self, expert_trajs: Dict[str, List[np.ndarray]]):
        self.expert_trajs = expert_trajs  # expert_id -> list of (T, D) np.ndarrays
        # Infer dimensionality & stats
        lengths = [traj.shape[0] for lst in expert_trajs.values() for traj in lst]
        self.num_experts = len(expert_trajs)
        self.total_trajs = sum(len(v) for v in expert_trajs.values())
        self.state_dim = next(iter(expert_trajs.values()))[0].shape[1] if self.total_trajs else 0
        self.lengths = lengths

    def length_stats(self):
        arr = np.array(self.lengths)
        return {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


def _standardize(raw: Any) -> Dict[str, List[np.ndarray]]:
    """Convert raw loaded pickle object into Dict[str, List[np.ndarray]].
    Expected raw structure: dict[expert_id] -> list/array of trajectories; each trajectory is array-like (T, D).
    """
    if not isinstance(raw, dict):
        raise ValueError("Top-level dataset must be a dict of experts.")
    expert_trajs: Dict[str, List[np.ndarray]] = {}
    for k, traj_list in raw.items():
        # Accept integer or string keys; normalize to string
        expert_id = str(k)
        if not isinstance(traj_list, (list, tuple)):
            raise ValueError(f"Expert {expert_id} value must be list-like of trajectories.")
        converted: List[np.ndarray] = []
        for i, t in enumerate(traj_list):
            arr = np.asarray(t)
            if arr.ndim != 2:
                raise ValueError(f"Trajectory {i} for expert {expert_id} must be 2D (T,D). Got shape {arr.shape}")
            converted.append(arr)
        if converted:
            expert_trajs[expert_id] = converted
    if not expert_trajs:
        raise ValueError("No trajectories found after processing.")
    # Validate uniform state dim
    dims = {traj.shape[1] for lst in expert_trajs.values() for traj in lst}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent state dimensions found: {dims}")
    return expert_trajs


def load_dataset(path: str | Path) -> LoadedDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        raw = pickle.load(f)
    expert_trajs = _standardize(raw)
    return LoadedDataset(expert_trajs)

__all__ = ["LoadedDataset", "load_dataset"]
