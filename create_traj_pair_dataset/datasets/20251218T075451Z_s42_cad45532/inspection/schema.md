# Sample Pair Schema
- pair_index: index within this inspection subset
- dataset_index: global pair index within the full dataset
- a_expert / b_expert: expert identifier strings
- a_traj_idx / b_traj_idx: indices within expert's trajectory list
- a_len / b_len: original trajectory lengths before padding/truncation
- label: 1 matching (same expert), 0 non-matching
