import pathlib
from datetime import datetime
import hashlib

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "datasets"
INSPECTION_SUBDIR = "inspection"


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def make_run_dir(base: pathlib.Path | None = None, seed: int | None = None, config_fingerprint: str | None = None) -> pathlib.Path:
    base = base or DEFAULT_DATASET_DIR
    base.mkdir(parents=True, exist_ok=True)
    parts = [timestamp()]
    if seed is not None:
        parts.append(f"s{seed}")
    if config_fingerprint:
        parts.append(config_fingerprint[:8])
    run_dir = base / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / INSPECTION_SUBDIR).mkdir(parents=True, exist_ok=True)
    return run_dir


def hash_config(cfg: dict) -> str:
    # Deterministic short hash of config (sorted keys)
    items = []
    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                flatten(f"{prefix}.{k}" if prefix else k, obj[k])
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                flatten(f"{prefix}[{i}]", v)
        else:
            items.append(f"{prefix}={obj}")
    flatten("", cfg)
    digest = hashlib.sha256("|".join(items).encode()).hexdigest()
    return digest

__all__ = [
    "ROOT",
    "DEFAULT_DATASET_DIR",
    "INSPECTION_SUBDIR",
    "timestamp",
    "make_run_dir",
    "hash_config",
]
