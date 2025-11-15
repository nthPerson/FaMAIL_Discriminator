from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
import random
import numpy as np


@dataclass
class TrajRef:
    expert_id: str
    traj_idx: int
    length: int
    mean_vec: np.ndarray


def build_catalog(expert_trajs: Dict[str, List[np.ndarray]]) -> List[TrajRef]:
    catalog: List[TrajRef] = []
    for eid, lst in expert_trajs.items():
        for i, arr in enumerate(lst):
            catalog.append(TrajRef(eid, i, arr.shape[0], arr.mean(axis=0)))
    return catalog


def sample_matching_pairs(expert_trajs: Dict[str, List[np.ndarray]], n_pairs: int, rng: random.Random,
                          strategy: str = "uniform") -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    experts = list(expert_trajs.keys())
    # Balanced expert sampling: rotate through experts
    exp_cycle = list(experts)
    rng.shuffle(exp_cycle)
    exp_pos = 0
    while len(pairs) < n_pairs:
        if not exp_cycle:
            exp_cycle = list(experts)
            rng.shuffle(exp_cycle)
        eid = exp_cycle[exp_pos % len(exp_cycle)]
        exp_pos += 1
        lst = expert_trajs[eid]
        if len(lst) < 2:
            continue
        i, j = rng.sample(range(len(lst)), 2)
        # Optionally extend strategy variations (e.g., length strata)
        pairs.append(((eid, i), (eid, j)))
    return pairs[:n_pairs]


def _semi_hard_selector(catalog: List[TrajRef], rng: random.Random, heuristic: str) -> Callable[[TrajRef], TrajRef]:
    # Precompute per-expert pools for efficiency
    by_expert: Dict[str, List[TrajRef]] = {}
    for ref in catalog:
        by_expert.setdefault(ref.expert_id, []).append(ref)

    def select(ref: TrajRef) -> TrajRef:
        candidates: List[TrajRef] = []
        for eid, lst in by_expert.items():
            if eid == ref.expert_id:
                continue
            candidates.extend(lst)
        if not candidates:
            raise RuntimeError("No candidates for semi-hard negatives")
        if heuristic == "length_diff":
            diffs = [abs(ref.length - c.length) for c in candidates]
            # Prefer smaller diffs: pick among K nearest
            K = min(50, len(candidates))
            nearest_idx = np.argpartition(diffs, K - 1)[:K]
            c = candidates[int(rng.choice(nearest_idx))]
            return c
        elif heuristic == "mean_cosine":
            # High cosine similarity (harder) across different experts
            ref_norm = np.linalg.norm(ref.mean_vec) + 1e-9
            sims = [float(np.dot(ref.mean_vec, c.mean_vec) / (ref_norm * (np.linalg.norm(c.mean_vec) + 1e-9))) for c in candidates]
            K = min(50, len(candidates))
            top_idx = np.argpartition(sims, len(sims) - K)[-K:]
            c = candidates[int(rng.choice(top_idx))]
            return c
        else:
            return rng.choice(candidates)

    return select


def sample_nonmatching_pairs(expert_trajs: Dict[str, List[np.ndarray]], n_pairs: int, rng: random.Random,
                             strategy: str = "uniform", semi_hard_heuristic: str = "length_diff") -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    catalog = build_catalog(expert_trajs)
    select_semi = _semi_hard_selector(catalog, rng, semi_hard_heuristic)

    # Balanced starting points across experts
    experts = list(expert_trajs.keys())
    rng.shuffle(experts)
    cursor = 0
    while len(pairs) < n_pairs:
        eid = experts[cursor % len(experts)]
        cursor += 1
        lst = expert_trajs[eid]
        if not lst:
            continue
        i = rng.randrange(len(lst))
        a = TrajRef(eid, i, lst[i].shape[0], lst[i].mean(axis=0))
        if strategy == "uniform":
            # pick random different expert trajectory
            other_eid = rng.choice([x for x in experts if x != eid])
            j = rng.randrange(len(expert_trajs[other_eid]))
            pairs.append(((eid, i), (other_eid, j)))
        elif strategy == "length_matched":
            # choose trajectory from other expert with closest length
            candidates = [c for c in catalog if c.expert_id != eid]
            if not candidates:
                continue
            diffs = [abs(a.length - c.length) for c in candidates]
            jidx = int(np.argmin(diffs))
            b = candidates[jidx]
            pairs.append(((eid, i), (b.expert_id, b.traj_idx)))
        elif strategy == "semi_hard":
            b = select_semi(a)
            pairs.append(((eid, i), (b.expert_id, b.traj_idx)))
        else:
            # default to uniform
            other_eid = rng.choice([x for x in experts if x != eid])
            j = rng.randrange(len(expert_trajs[other_eid]))
            pairs.append(((eid, i), (other_eid, j)))
    return pairs[:n_pairs]

__all__ = [
    "TrajRef",
    "build_catalog",
    "sample_matching_pairs",
    "sample_nonmatching_pairs",
]
