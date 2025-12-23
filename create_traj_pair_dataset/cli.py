#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random

import numpy as np

from .data_loader import load_dataset
from .pair_sampling import sample_matching_pairs, sample_nonmatching_pairs
from .assembly import assemble_pairs
from .serialization import save_arrays_npz, save_torch, write_meta, write_report, write_inspection
from .paths import make_run_dir, hash_config


def parse_args():
    p = argparse.ArgumentParser("HuMID pair dataset generator")
    p.add_argument("--data", type=str, required=True, help="Path to all_trajs.pkl")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default discriminator/datasets)")
    p.add_argument("--n-pos", type=int, default=None, help="Number of matching pairs")
    p.add_argument("--n-neg", type=int, default=None, help="Number of non-matching pairs")
    p.add_argument("--strategy-pos", type=str, default=None, choices=["uniform"], help="Positive sampling strategy")
    p.add_argument("--strategy-neg", type=str, default=None, choices=["uniform", "length_matched", "semi_hard"], help="Negative sampling strategy")
    p.add_argument("--semi-hard-heuristic", type=str, default=None, choices=["length_diff", "mean_cosine"], help="Heuristic for semi-hard negatives")
    p.add_argument("--percentile-len", type=int, default=None, help="Percentile for target length padding/truncation")
    p.add_argument("--pad-mode", type=str, default=None, choices=["right", "center"], help="Padding mode")
    p.add_argument("--truncate-mode", type=str, default=None, choices=["tail", "head", "center"], help="Truncation mode when length exceeds target")
    p.add_argument("--state-start", type=int, default=None, help="Inclusive start index for state dimension slice")
    p.add_argument("--state-end", type=int, default=None, help="Exclusive end index for state dimension slice")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--mini", action="store_true", help="Generate a small representative sample dataset")
    p.add_argument("--inspect-k", type=int, default=50, help="Number of pairs to include in inspection JSON")
    p.add_argument("--config-defaults", type=str, default=str(Path(__file__).resolve().parent / "config_defaults.json"), help="Path to defaults JSON")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Load defaults
    with open(args.config_defaults, "r") as f:
        defaults = json.load(f)

    # Resolve effective config
    eff = {
        "n_pos": args.n_pos if args.n_pos is not None else (defaults["mini_n_pos"] if args.mini else defaults["n_pos"]),
        "n_neg": args.n_neg if args.n_neg is not None else (defaults["mini_n_neg"] if args.mini else defaults["n_neg"]),
        "strategy_pos": args.strategy_pos or defaults["strategy_pos"],
        "strategy_neg": args.strategy_neg or defaults["strategy_neg"],
        "semi_hard_heuristic": args.semi_hard_heuristic or defaults["semi_hard_heuristic"],
        "percentile_len": args.percentile_len if args.percentile_len is not None else defaults["percentile_len"],
        "pad_mode": args.pad_mode or defaults["pad_mode"],
        "truncate_mode": args.truncate_mode or defaults["truncate_mode"],
        "state_start": args.state_start if args.state_start is not None else defaults.get("state_start"),
        "state_end": args.state_end if args.state_end is not None else defaults.get("state_end"),
        "mini": bool(args.mini),
        "inspect_k": int(args.inspect_k),
        "seed": int(args.seed),
    }

    # Prepare output dir
    base_out = Path(args.out_dir) if args.out_dir else None
    fingerprint = hash_config(eff)
    out_dir = make_run_dir(base_out, seed=eff["seed"], config_fingerprint=fingerprint)

    # Load data
    ds = load_dataset(args.data)

    # Optional state-dimension slicing
    def slice_state_dims(expert_trajs, start, end):
        sample_traj = next(iter(expert_trajs.values()))[0]
        D = sample_traj.shape[1]
        s = 0 if start is None else start
        e = D if end is None else end
        if s < 0 or e < 1 or s >= e or e > D:
            raise ValueError(f"Invalid state slice [{s}:{e}] for state dimension {D}")
        sliced = {}
        for eid, lst in expert_trajs.items():
            sliced[eid] = [traj[:, s:e] for traj in lst]
        return sliced

    # Optionally reduce per-expert trajectories in mini mode
    expert_trajs = ds.expert_trajs
    expert_trajs = slice_state_dims(expert_trajs, eff["state_start"], eff["state_end"])
    if eff["mini"]:
        reduced = {}
        max_per = defaults.get("mini_max_traj_per_expert", 2)
        for eid, lst in expert_trajs.items():
            idxs = list(range(len(lst)))
            rng.shuffle(idxs)
            keep = [lst[i] for i in idxs[:min(max_per, len(lst))]]
            if len(keep) >= 1:
                reduced[eid] = keep
        expert_trajs = reduced

    # Sample pairs
    pos_pairs = sample_matching_pairs(expert_trajs, eff["n_pos"], rng, strategy=eff["strategy_pos"])  # same expert
    neg_pairs = sample_nonmatching_pairs(expert_trajs, eff["n_neg"], rng, strategy=eff["strategy_neg"], semi_hard_heuristic=eff["semi_hard_heuristic"])  # different experts

    # Assemble arrays
    arrays = assemble_pairs(expert_trajs, pos_pairs, neg_pairs, eff["percentile_len"], eff["pad_mode"], eff["truncate_mode"])

    # Save outputs
    save_arrays_npz(out_dir, arrays)
    save_torch(out_dir, arrays)
    write_meta(out_dir, eff, arrays, fingerprint, eff["seed"]) 
    write_report(out_dir, arrays, arrays["meta"]) 
    write_inspection(out_dir, arrays, arrays["meta"], eff["inspect_k"], eff["seed"])

    print(f"Wrote dataset to {out_dir}")


if __name__ == "__main__":
    main()
