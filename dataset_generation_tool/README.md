# Trajectory Pair Dataset Generator

Streamlit-based tool for producing labeled trajectory pairs for the ST-SiameseNet discriminator. Implements the requirements from the "Trajectory Pair Dataset Generator" design doc.

## Quick start
```bash
pip install -r discriminator/dataset_generation_tool/requirements.txt
streamlit run discriminator/dataset_generation_tool/app.py
```

The sidebar lets you set:
- Positive/negative pair counts and random seed
- Agent sampling distribution (proportional or uniform)
- Positive strategy (random vs sequential non-overlapping segments)
- Negative strategy (random vs round-robin agents)
- Concatenated trajectory span (1–7 days)
- Optional extra feature slice beyond indices 0–3 (defaults to none)
- Padding/truncation mode (defaults to truncate to shorter; pad or fixed length also available)

## Data assumptions
- Input pickle (`all_trajs.pkl`) is a `dict` of 50 agents → list of trajectories.
- Each trajectory is `(T, 126)`; indices 0–3 are `x_grid, y_grid, time_bucket, day_index`.
- Trajectories per agent are assumed to be in chronological order.

## Outputs
- Arrays `x1, x2, mask1, mask2, label` (labels: 0 = same agent, 1 = different agents).
- Metadata JSON with config, hash, length stats, and per-agent usage counts.
- Download buttons for `.npz`, `.pt` (if PyTorch installed), and a small JSON sample.

## Notes
- Positive pairs are built from non-overlapping segments of the same agent.
- Negative pairs draw segments from different agents; round-robin option broadens coverage.
- All generated sequences are padded to a uniform length for export convenience.
