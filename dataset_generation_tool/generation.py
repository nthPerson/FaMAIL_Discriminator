"""Core generation logic for trajectory pair datasets.

Implements loading, segment construction, sampling, alignment, and metadata
construction for the Streamlit UI defined in app.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import io
import json
import random

import numpy as np


DAY_BUCKETS = 288


@dataclass
class Segment:
    agent_id: str
    start_time: float
    end_time: float
    traj_indices: List[int]
    component_lengths: List[int]
    data: np.ndarray

    @property
    def length(self) -> int:
        return int(self.data.shape[0])


@dataclass
class GenerationConfig:
    data_path: Path
    positive_pairs: int
    negative_pairs: int
    days: int = 2
    feature_start: int = 4
    feature_end: int = 4  # default: no extra features beyond indices 0-3
    padding: str = "truncate_to_shorter"  # pad_to_longer | truncate_to_shorter | fixed_length
    fixed_length: Optional[int] = None
    positive_strategy: str = "random"  # random | sequential
    negative_strategy: str = "random"  # random | round_robin
    agent_distribution: str = "proportional"  # proportional | uniform
    seed: Optional[int] = None
    ensure_agent_coverage: bool = True

    def clamped_feature_bounds(self, state_dim: int) -> Tuple[int, int]:
        start = max(4, min(self.feature_start, state_dim))
        end = max(start, min(self.feature_end, state_dim))
        return start, end


def _compute_global_time(arr: np.ndarray) -> np.ndarray:
    days = arr[:, 3]
    buckets = arr[:, 2]
    return (days - 1) * DAY_BUCKETS + buckets


def load_dataset(path: Path) -> Dict[str, List[np.ndarray]]:
    import pickle

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Top-level dataset must be a dict of expert_id -> trajectories")
    expert_trajs: Dict[str, List[np.ndarray]] = {}
    for k, traj_list in raw.items():
        expert_id = str(k)
        if not isinstance(traj_list, (list, tuple)):
            raise ValueError(f"Expert {expert_id} payload must be list-like")
        converted: List[np.ndarray] = []
        for i, t in enumerate(traj_list):
            arr = np.asarray(t)
            if arr.ndim != 2:
                raise ValueError(f"Trajectory {i} for expert {expert_id} must be 2D")
            converted.append(arr)
        if converted:
            expert_trajs[agent_id_or_default(expert_id)] = converted
    if not expert_trajs:
        raise ValueError("No trajectories after conversion")
    dims = {traj.shape[1] for lst in expert_trajs.values() for traj in lst}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent state dims found: {dims}")
    return expert_trajs


def agent_id_or_default(eid: str) -> str:
    return eid if eid else "unknown"


def build_segments(expert_trajs: Dict[str, List[np.ndarray]], days: int) -> Dict[str, List[Segment]]:
    segments: Dict[str, List[Segment]] = {}
    for agent_id, trajs in expert_trajs.items():
        segs_for_agent: List[Segment] = []
        n = len(trajs)
        if n == 0:
            continue
        for start_idx in range(n):
            seg_arrays: List[np.ndarray] = []
            seg_traj_indices: List[int] = []
            comp_lengths: List[int] = []
            start_time: Optional[float] = None
            last_time: Optional[float] = None
            for cursor in range(start_idx, n):
                arr = trajs[cursor]
                times = _compute_global_time(arr)
                if start_time is None:
                    start_time = float(times[0])
                if last_time is not None and times[0] <= last_time:
                    break
                seg_arrays.append(arr)
                seg_traj_indices.append(cursor)
                comp_lengths.append(int(arr.shape[0]))
                last_time = float(times[-1])
                span_days = (last_time - start_time) / DAY_BUCKETS
                if span_days >= days:
                    data = np.concatenate(seg_arrays, axis=0)
                    segs_for_agent.append(
                        Segment(
                            agent_id=agent_id,
                            start_time=start_time,
                            end_time=last_time,
                            traj_indices=list(seg_traj_indices),
                            component_lengths=list(comp_lengths),
                            data=data,
                        )
                    )
                    break
                if cursor == n - 1:
                    data = np.concatenate(seg_arrays, axis=0)
                    segs_for_agent.append(
                        Segment(
                            agent_id=agent_id,
                            start_time=start_time,
                            end_time=last_time,
                            traj_indices=list(seg_traj_indices),
                            component_lengths=list(comp_lengths),
                            data=data,
                        )
                    )
        if segs_for_agent:
            segments[agent_id] = segs_for_agent
    return segments


def _non_overlapping(a: Segment, b: Segment) -> bool:
    return a.end_time < b.start_time or b.end_time < a.start_time


def _pick_positive_pair(segs: List[Segment], rng: random.Random, strategy: str) -> Optional[Tuple[Segment, Segment]]:
    if len(segs) < 2:
        return None
    attempts = 0
    segs_sorted = sorted(segs, key=lambda s: s.start_time)
    while attempts < 20:
        if strategy == "sequential":
            first = rng.choice(segs_sorted)
            candidates = [s for s in segs_sorted if _non_overlapping(first, s)]
            if not candidates:
                attempts += 1
                continue
            second = max(candidates, key=lambda s: abs(s.start_time - first.start_time))
        else:
            first, second = rng.sample(segs, 2)
            if not _non_overlapping(first, second):
                attempts += 1
                continue
        if _non_overlapping(first, second):
            return first, second
        attempts += 1
    return None


def _agent_weights(segments: Dict[str, List[Segment]], mode: str) -> List[float]:
    if mode == "uniform":
        return [1.0 for _ in segments]
    weights = []
    for segs in segments.values():
        weights.append(float(sum(s.length for s in segs)))
    return weights


def sample_positive_pairs(
    segments: Dict[str, List[Segment]],
    n_pairs: int,
    rng: random.Random,
    strategy: str,
    distribution: str,
    ensure_coverage: bool,
) -> List[Tuple[Segment, Segment]]:
    pairs: List[Tuple[Segment, Segment]] = []
    agents = [a for a, segs in segments.items() if len(segs) >= 2]
    if not agents:
        return pairs
    weights = _agent_weights({a: segments[a] for a in agents}, distribution)
    if ensure_coverage:
        for agent_id in agents:
            if len(pairs) >= n_pairs:
                break
            pair = _pick_positive_pair(segments[agent_id], rng, strategy)
            if pair:
                pairs.append(pair)
    attempts = 0
    max_attempts = n_pairs * 10 + 200
    while len(pairs) < n_pairs and attempts < max_attempts:
        agent_id = rng.choices(agents, weights=weights, k=1)[0]
        pair = _pick_positive_pair(segments[agent_id], rng, strategy)
        if pair:
            pairs.append(pair)
        attempts += 1
    return pairs[:n_pairs]


def _pick_negative_pair(
    segments: Dict[str, List[Segment]],
    rng: random.Random,
    distribution: str,
    negative_strategy: str,
    anchor_agent: Optional[str] = None,
) -> Optional[Tuple[Segment, Segment]]:
    agents = list(segments.keys())
    if len(agents) < 2:
        return None
    weights = _agent_weights(segments, distribution)
    if anchor_agent is None:
        anchor_agent = rng.choices(agents, weights=weights, k=1)[0]
    anchor_segs = segments.get(anchor_agent, [])
    if not anchor_segs:
        return None
    other_agents = [a for a in agents if a != anchor_agent and segments.get(a)]
    if not other_agents:
        return None
    if negative_strategy == "round_robin":
        other_agents = sorted(other_agents)
    other_agent = rng.choice(other_agents)
    b_segs = segments[other_agent]
    seg_a = rng.choice(anchor_segs)
    seg_b = rng.choice(b_segs)
    return seg_a, seg_b


def sample_negative_pairs(
    segments: Dict[str, List[Segment]],
    n_pairs: int,
    rng: random.Random,
    strategy: str,
    distribution: str,
    ensure_coverage: bool,
) -> List[Tuple[Segment, Segment]]:
    pairs: List[Tuple[Segment, Segment]] = []
    agents = list(segments.keys())
    if len(agents) < 2:
        return pairs
    if ensure_coverage:
        for agent_id in agents:
            if len(pairs) >= n_pairs:
                break
            pair = _pick_negative_pair(segments, rng, distribution, strategy, anchor_agent=agent_id)
            if pair:
                pairs.append(pair)
    attempts = 0
    max_attempts = n_pairs * 10 + 200
    while len(pairs) < n_pairs and attempts < max_attempts:
        pair = _pick_negative_pair(segments, rng, distribution, strategy)
        if pair:
            pairs.append(pair)
        attempts += 1
    return pairs[:n_pairs]


def _slice_features(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    base = arr[:, :4]
    sliced = arr[:, start:end]
    return np.concatenate([base, sliced], axis=1)


def align_pair(
    seq1: np.ndarray,
    seq2: np.ndarray,
    mode: str,
    fixed_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    len1, len2 = seq1.shape[0], seq2.shape[0]
    if mode == "truncate_to_shorter":
        target = min(len1, len2)
        seq1 = seq1[:target]
        seq2 = seq2[:target]
        mask1 = np.ones(target, dtype=np.int32)
        mask2 = np.ones(target, dtype=np.int32)
        return seq1, seq2, mask1, mask2, target
    if mode == "fixed_length" and fixed_length:
        target = fixed_length
    else:
        target = max(len1, len2)
    pad1 = target - len1
    pad2 = target - len2
    if pad1 > 0:
        seq1 = np.pad(seq1, ((0, pad1), (0, 0)), mode="constant", constant_values=0.0)
    else:
        seq1 = seq1[:target]
    if pad2 > 0:
        seq2 = np.pad(seq2, ((0, pad2), (0, 0)), mode="constant", constant_values=0.0)
    else:
        seq2 = seq2[:target]
    mask1 = np.zeros(target, dtype=np.int32)
    mask1[: min(len1, target)] = 1
    mask2 = np.zeros(target, dtype=np.int32)
    mask2[: min(len2, target)] = 1
    return seq1, seq2, mask1, mask2, target


def assemble_dataset(
    config: GenerationConfig,
    preview_only: bool = False,
    preview_cap: int = 12,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    rng = random.Random(config.seed)
    expert_trajs = load_dataset(config.data_path)
    state_dim = next(iter(expert_trajs.values()))[0].shape[1]
    feat_start, feat_end = config.clamped_feature_bounds(state_dim)
    segments = build_segments(expert_trajs, config.days)
    pos_pairs = sample_positive_pairs(
        segments,
        n_pairs=config.positive_pairs if not preview_only else min(config.positive_pairs, preview_cap),
        rng=rng,
        strategy=config.positive_strategy,
        distribution=config.agent_distribution,
        ensure_coverage=config.ensure_agent_coverage,
    )
    neg_pairs = sample_negative_pairs(
        segments,
        n_pairs=config.negative_pairs if not preview_only else min(config.negative_pairs, preview_cap),
        rng=rng,
        strategy=config.negative_strategy,
        distribution=config.agent_distribution,
        ensure_coverage=config.ensure_agent_coverage,
    )
    pairs = [(p[0], p[1], 0) for p in pos_pairs] + [(p[0], p[1], 1) for p in neg_pairs]
    rng.shuffle(pairs)
    sequences1: List[np.ndarray] = []
    sequences2: List[np.ndarray] = []
    masks1: List[np.ndarray] = []
    masks2: List[np.ndarray] = []
    labels: List[int] = []
    lengths_raw: List[Tuple[int, int]] = []
    agent_usage: Dict[str, Dict[str, int]] = {}
    pair_info: List[Dict[str, Any]] = []
    for seg_a, seg_b, label in pairs:
        seq1 = _slice_features(seg_a.data, feat_start, feat_end).astype(np.float32)
        seq2 = _slice_features(seg_b.data, feat_start, feat_end).astype(np.float32)
        aligned1, aligned2, mask1, mask2, target_len = align_pair(seq1, seq2, config.padding, config.fixed_length)
        sequences1.append(aligned1)
        sequences2.append(aligned2)
        masks1.append(mask1)
        masks2.append(mask2)
        labels.append(label)
        lengths_raw.append((seq1.shape[0], seq2.shape[0]))
        agent_usage.setdefault(seg_a.agent_id, {"pos": 0, "neg": 0})
        agent_usage.setdefault(seg_b.agent_id, {"pos": 0, "neg": 0})
        if label == 0:
            agent_usage[seg_a.agent_id]["pos"] += 1
            agent_usage[seg_b.agent_id]["pos"] += 1
        else:
            agent_usage[seg_a.agent_id]["neg"] += 1
            agent_usage[seg_b.agent_id]["neg"] += 1
        pair_info.append(
            {
                "agent_a": seg_a.agent_id,
                "agent_b": seg_b.agent_id,
                "label": int(label),
                "len_raw_a": int(seq1.shape[0]),
                "len_raw_b": int(seq2.shape[0]),
                "align_len": int(target_len),
                "traj_indices_a": list(seg_a.traj_indices),
                "traj_indices_b": list(seg_b.traj_indices),
                "component_lengths_a": list(seg_a.component_lengths),
                "component_lengths_b": list(seg_b.component_lengths),
                "start_time_a": seg_a.start_time,
                "start_time_b": seg_b.start_time,
                "end_time_a": seg_a.end_time,
                "end_time_b": seg_b.end_time,
            }
        )
    if not sequences1:
        raise RuntimeError("No pairs generated. Check data availability and settings.")
    max_len = max(seq.shape[0] for seq in sequences1)
    # Ensure uniform length across dataset for saving convenience
    def pad_to_length(arr_list: List[np.ndarray]) -> np.ndarray:
        out = []
        for arr in arr_list:
            if arr.shape[0] == max_len:
                out.append(arr)
                continue
            pad = max_len - arr.shape[0]
            out.append(np.pad(arr, ((0, pad), (0, 0)), mode="constant", constant_values=0.0))
        return np.stack(out, axis=0)

    def pad_masks(mask_list: List[np.ndarray]) -> np.ndarray:
        out = []
        for mask in mask_list:
            if mask.shape[0] == max_len:
                out.append(mask)
                continue
            pad = max_len - mask.shape[0]
            out.append(np.pad(mask, (0, pad), mode="constant", constant_values=0))
        return np.stack(out, axis=0)

    x1 = pad_to_length(sequences1)
    x2 = pad_to_length(sequences2)
    mask1_arr = pad_masks(masks1)
    mask2_arr = pad_masks(masks2)
    label_arr = np.array(labels, dtype=np.int64)

    dataset = {
        "x1": x1,
        "x2": x2,
        "mask1": mask1_arr,
        "mask2": mask2_arr,
        "label": label_arr,
    }
    metadata = _build_metadata(
        config=config,
        feat_start=feat_start,
        feat_end=feat_end,
        max_len=max_len,
        lengths_raw=lengths_raw,
        agent_usage=agent_usage,
        total_pairs=len(pairs),
        pos_pairs=len(pos_pairs),
        neg_pairs=len(neg_pairs),
    )
    return dataset, metadata, pair_info


def _length_stats(lengths: List[int]) -> Dict[str, float]:
    arr = np.array(lengths, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _build_metadata(
    config: GenerationConfig,
    feat_start: int,
    feat_end: int,
    max_len: int,
    lengths_raw: List[Tuple[int, int]],
    agent_usage: Dict[str, Dict[str, int]],
    total_pairs: int,
    pos_pairs: int,
    neg_pairs: int,
) -> Dict[str, Any]:
    lens1 = [a for a, _ in lengths_raw]
    lens2 = [b for _, b in lengths_raw]
    combined = lens1 + lens2
    cfg_dict = asdict(config)
    cfg_dict["feature_start"] = feat_start
    cfg_dict["feature_end"] = feat_end
    cfg_dict["data_path"] = str(config.data_path)
    hash_payload = json.dumps(cfg_dict, sort_keys=True).encode()
    dataset_hash = hashlib.sha256(hash_payload).hexdigest()[:12]
    return {
        "config": cfg_dict,
        "counts": {
            "total_pairs": total_pairs,
            "positive_pairs": pos_pairs,
            "negative_pairs": neg_pairs,
        },
        "length_stats": {
            "x1": _length_stats(lens1),
            "x2": _length_stats(lens2),
            "combined": _length_stats(combined),
            "padded_length": max_len,
        },
        "agent_usage": agent_usage,
        "dataset_hash": dataset_hash,
    }


def dataset_to_npz_bytes(dataset: Dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **dataset)
    return buf.getvalue()


def dataset_to_pt_bytes(dataset: Dict[str, np.ndarray]) -> bytes:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for .pt export") from exc
    buf = io.BytesIO()
    torch.save({k: torch.as_tensor(v) for k, v in dataset.items()}, buf)
    return buf.getvalue()


def sample_json(dataset: Dict[str, np.ndarray], metadata: Dict[str, any], k: int = 5) -> str:
    total = dataset["label"].shape[0]
    k = min(k, total)
    idx = list(range(total))
    random.shuffle(idx)
    idx = idx[:k]
    sample = []
    for i in idx:
        sample.append(
            {
                "label": int(dataset["label"][i]),
                "len_x1": int(dataset["mask1"][i].sum()),
                "len_x2": int(dataset["mask2"][i].sum()),
            }
        )
    return json.dumps({"sample_pairs": sample, "metadata": metadata}, indent=2)
