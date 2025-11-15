from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from .paths import INSPECTION_SUBDIR

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


def save_arrays_npz(out_dir: Path, arrays: Dict[str, Any]):
    npz_path = out_dir / "pairs.npz"
    np.savez_compressed(npz_path, **{k: v for k, v in arrays.items() if k not in {"meta"}})
    return npz_path


def save_torch(out_dir: Path, arrays: Dict[str, Any]):
    if torch is None:
        return None
    tensor_dict = {}
    for k, v in arrays.items():
        if k == "meta":
            continue
        if isinstance(v, np.ndarray):
            tensor_dict[k] = torch.from_numpy(v)
    pt_path = out_dir / "pairs.pt"
    torch.save(tensor_dict, pt_path)
    return pt_path


def write_meta(out_dir: Path, config: Dict[str, Any], arrays: Dict[str, Any], fingerprint: str, seed: int):
    meta_path = out_dir / "meta.json"
    meta = {
        "config": config,
        "seed": seed,
        "fingerprint": fingerprint,
        "counts": {
            "total_pairs": int(arrays["label"].shape[0]),
            "positives": int(int(arrays["label"].sum())),
            "negatives": int(int(arrays["label"].shape[0] - arrays["label"].sum())),
        },
        "target_len": int(arrays["x1"].shape[1]),
        "state_dim": int(arrays["x1"].shape[2]),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def write_report(out_dir: Path, arrays: Dict[str, Any], meta_objects: List[Any]):
    report = []
    lengths_pos = []
    lengths_neg = []
    for m in meta_objects:
        if m.label == 1:
            lengths_pos.append((m.a_len + m.b_len) / 2)
        else:
            lengths_neg.append((m.a_len + m.b_len) / 2)
    import numpy as np
    def stats(xs):
        if not xs:
            return "n=0"
        arr = np.array(xs)
        return f"n={arr.size} mean={arr.mean():.1f} median={np.median(arr):.1f} p90={np.percentile(arr,90):.1f} p95={np.percentile(arr,95):.1f}"
    report.append("# HuMID Pair Dataset Report\n")
    report.append("## Length Stats\n")
    report.append(f"Positives: {stats(lengths_pos)}\n")
    report.append(f"Negatives: {stats(lengths_neg)}\n")
    path = out_dir / "report.md"
    path.write_text("\n".join(report))
    return path


def write_inspection(out_dir: Path, arrays: Dict[str, Any], meta_objects: List[Any], k: int):
    insp_dir = out_dir / INSPECTION_SUBDIR
    insp_dir.mkdir(exist_ok=True, parents=True)
    k = min(k, len(meta_objects))
    if k <= 0:
        return None
    # Simple selection: first k (could randomize upstream)
    sample_meta = meta_objects[:k]
    export = []
    for idx, m in enumerate(sample_meta):
        export.append({
            "pair_index": idx,
            "a_expert": m.a_expert,
            "a_traj_idx": m.a_idx,
            "b_expert": m.b_expert,
            "b_traj_idx": m.b_idx,
            "a_len": m.a_len,
            "b_len": m.b_len,
            "label": m.label,
        })
    schema_md = insp_dir / "schema.md"
    schema_md.write_text("""# Sample Pair Schema\n- pair_index: index in exported selection\n- a_expert / b_expert: expert identifier strings\n- a_traj_idx / b_traj_idx: indices within expert's trajectory list\n- a_len / b_len: original trajectory lengths before padding/truncation\n- label: 1 matching (same expert), 0 non-matching\n""")
    sample_json = insp_dir / "sample_pairs.json"
    sample_json.write_text(json.dumps(export, indent=2))
    index_json = insp_dir / "index.json"
    index_json.write_text(json.dumps({
        "sample_pairs": sample_json.name,
        "schema": schema_md.name,
        "count": k,
    }, indent=2))
    return sample_json

__all__ = [
    "save_arrays_npz",
    "save_torch",
    "write_meta",
    "write_report",
    "write_inspection",
]
