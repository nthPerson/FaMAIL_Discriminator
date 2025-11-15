# HuMID Trajectory Pair Dataset Pipeline

Generates matching (same expert) and non-matching (different expert) trajectory pairs from `all_trajs.pkl` for training a binary discriminator (HuMID) model.

## Features
- Supports full and mini (representative) dataset modes.
- Positive & negative pair count tuning (`--n-pos`, `--n-neg`).
- Multiple negative sampling strategies: `uniform`, `length_matched`, `semi_hard`.
- Semi-hard heuristic choices: `length_diff` (closest lengths) or `mean_cosine` (similar mean state vectors).
- Deterministic, seed-based reproducibility with config fingerprint & metadata.
- Padding & truncation configurable by percentile length, mode, and strategy.
- Dual output formats: compressed NumPy (`pairs.npz`) and Torch (`pairs.pt`).
- Rich metadata (`meta.json`) and quick stats report (`report.md`).
- Human inspection subset (`inspection/sample_pairs.json` + schema) for qualitative review.
- Streamlit app (`streamlit_app.py`) for interactive exploration (stats, pair filtering, PCA visualization).

## Data Assumptions
`all_trajs.pkl` structure:
```
{
  expert_id_0: [ np.ndarray(shape=(T0, 126)), np.ndarray(shape=(T1, 126)), ... ],
  expert_id_1: [ ... ],
  ... (≈50 experts)
}
```
All trajectories share the same state dimension (126).

## Installation (Environment Assumes PyTorch Installed)
If PyTorch not already installed:
```bash
pip install torch
```
(No extra dependencies beyond standard library + NumPy.)

## Basic Usage
From repository root:
```bash
python -m create_traj_pair_dataset.cli --data create_traj_pair_dataset/source_data/all_trajs.pkl --n-pos 8000 --n-neg 8000
```
Outputs to `create_traj_pair_dataset/datasets/<timestamp>_s<seed>_<hash>/`.

## Mini Mode (Fast Sanity Check)
```bash
python -m create_traj_pair_dataset.cli --data create_traj_pair_dataset/source_data/all_trajs.pkl --mini --inspect-k 30
```
Mini mode automatically reduces per-expert trajectories (default 2) and pair counts (see `config_defaults.json`).

## Key Arguments
| Argument | Description | Default (full) |
|----------|-------------|----------------|
| `--n-pos` | Number of matching pairs | 5000 |
| `--n-neg` | Number of non-matching pairs | 5000 |
| `--strategy-pos` | Positive pair strategy | uniform |
| `--strategy-neg` | Negative pair strategy | uniform |
| `--semi-hard-heuristic` | Heuristic for semi-hard negatives | length_diff |
| `--percentile-len` | Target length percentile for padding/truncation | 95 |
| `--pad-mode` | Padding placement | right |
| `--truncate-mode` | Truncation slice | tail |
| `--inspect-k` | Pairs exported for inspection | 50 |
| `--mini` | Enable representative downsized mode | off |
| `--seed` | RNG seed | 42 |

See `config_defaults.json` for mini-mode overrides.

## Output Structure
```
<run_dir>/
  pairs.npz          # x1, x2, mask1, mask2, label arrays
  pairs.pt           # Torch tensor dict (same keys)
  meta.json          # config + counts + dimensions
  report.md          # basic statistics
  inspection/
    sample_pairs.json
    schema.md
    index.json
```

## Loading Example (PyTorch)
```python
import torch, numpy as np
import numpy as np
import torch
import json
from pathlib import Path

run_dir = Path('create_traj_pair_dataset/datasets/...')
# Torch tensors
pairs = torch.load(run_dir / 'pairs.pt')
print(pairs['x1'].shape, pairs['label'].shape)

# Or NumPy
npz = np.load(run_dir / 'pairs.npz')
print(npz['x1'].shape, npz['label'].shape)

# Metadata
meta = json.loads((run_dir / 'meta.json').read_text())
print(meta['counts'])
```

## Streamlit Inspection UI
Launch the interactive inspector (requires `streamlit`, `plotly`, and `scikit-learn` for PCA):
```bash
pip install streamlit plotly scikit-learn
streamlit run create_traj_pair_dataset/streamlit_app.py -- --dataset create_traj_pair_dataset/datasets/<run_dir>
```
If `--dataset` is omitted, you can pick a run from a dropdown.

Features:
- Overview tab: summary metrics + length histograms and percentile stats.
- Pair tab: select / filter by label, view trajectory lengths, mean feature cosine, optional PCA 2D projection, raw snippet preview.
- Report tab: renders `report.md`.

Use the sidebar to filter positives/negatives, select index numerically or choose a random pair.

## Negative Sampling Strategy Notes
- `uniform`: random different-expert pairing.
- `length_matched`: chooses other-expert trajectory closest in length.
- `semi_hard`: uses heuristic (`length_diff` or `mean_cosine`) to bias toward harder (similar) negatives.
  - `length_diff`: selects among a small set of nearest lengths.
  - `mean_cosine`: selects among a small set of highest cosine similarity mean feature vectors.

## Reproducibility
Each run directory name embeds UTC timestamp, seed (`s<seed>`), and an 8-char config hash. Full config stored in `meta.json`.

## Extending
- Add a Parquet manifest for large-scale sharding.
- Add additional semi-hard heuristics (e.g., DTW distance) in `pair_sampling.py`.
- Integrate a Streamlit visualization using `inspection/sample_pairs.json`.
 - Enhance Streamlit app with trajectory geographic plotting (if spatial dims known).

## License
Follows repository license.
