from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


@dataclass
class PairIndex:
    # references for metadata
    a_expert: str
    a_idx: int
    b_expert: str
    b_idx: int
    a_len: int
    b_len: int
    label: int  # 1=matching, 0=non-matching


def compute_target_len(lengths: List[int], percentile: int | None) -> int:
    arr = np.array(lengths)
    if percentile:
        return int(np.percentile(arr, percentile))
    return int(arr.max())


def pad_or_truncate(seq: np.ndarray, L: int, pad_mode: str = "right", truncate_mode: str = "tail") -> tuple[np.ndarray, np.ndarray]:
    # seq: (T, D)
    T, D = seq.shape
    if T == L:
        return seq, np.ones(L, dtype=np.uint8)
    if T > L:
        if truncate_mode == "tail":
            sliced = seq[:L]
        elif truncate_mode == "head":
            sliced = seq[-L:]
        elif truncate_mode == "center":
            start = (T - L) // 2
            sliced = seq[start:start+L]
        else:
            sliced = seq[:L]
        return sliced, np.ones(L, dtype=np.uint8)
    # T < L: pad
    pad_len = L - T
    pad_block = np.zeros((pad_len, D), dtype=seq.dtype)
    if pad_mode == "right":
        out = np.concatenate([seq, pad_block], axis=0)
        mask = np.concatenate([np.ones(T, dtype=np.uint8), np.zeros(pad_len, dtype=np.uint8)])
    elif pad_mode == "center":
        left = pad_len // 2
        right = pad_len - left
        out = np.concatenate([np.zeros((left, D), dtype=seq.dtype), seq, np.zeros((right, D), dtype=seq.dtype)], axis=0)
        mask = np.concatenate([np.zeros(left, dtype=np.uint8), np.ones(T, dtype=np.uint8), np.zeros(right, dtype=np.uint8)])
    else:
        out = np.concatenate([seq, pad_block], axis=0)
        mask = np.concatenate([np.ones(T, dtype=np.uint8), np.zeros(pad_len, dtype=np.uint8)])
    return out, mask


def assemble_pairs(expert_trajs: Dict[str, List[np.ndarray]],
                   pos_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                   neg_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
                   percentile_len: int | None,
                   pad_mode: str,
                   truncate_mode: str):
    lengths = [traj.shape[0] for lst in expert_trajs.values() for traj in lst]
    L = compute_target_len(lengths, percentile_len)
    D = next(iter(expert_trajs.values()))[0].shape[1]
    N = len(pos_pairs) + len(neg_pairs)

    x1 = np.zeros((N, L, D), dtype=np.float32)
    x2 = np.zeros((N, L, D), dtype=np.float32)
    m1 = np.zeros((N, L), dtype=np.uint8)
    m2 = np.zeros((N, L), dtype=np.uint8)
    y = np.zeros((N,), dtype=np.int64)
    meta: List[PairIndex] = []

    all_pairs = [(p, 1) for p in pos_pairs] + [(n, 0) for n in neg_pairs]

    for idx, (pair, label) in enumerate(all_pairs):
        (ae, ai), (be, bi) = pair
        a = expert_trajs[ae][ai]
        b = expert_trajs[be][bi]
        a_proc, am = pad_or_truncate(a, L, pad_mode=pad_mode, truncate_mode=truncate_mode)
        b_proc, bm = pad_or_truncate(b, L, pad_mode=pad_mode, truncate_mode=truncate_mode)
        x1[idx] = a_proc.astype(np.float32)
        x2[idx] = b_proc.astype(np.float32)
        m1[idx] = am
        m2[idx] = bm
        y[idx] = label
        meta.append(PairIndex(ae, ai, be, bi, a.shape[0], b.shape[0], label))

    return {
        "x1": x1,
        "x2": x2,
        "mask1": m1,
        "mask2": m2,
        "label": y,
        "meta": meta,
        "target_len": L,
        "state_dim": D,
    }

__all__ = [
    "PairIndex",
    "compute_target_len",
    "pad_or_truncate",
    "assemble_pairs",
]
