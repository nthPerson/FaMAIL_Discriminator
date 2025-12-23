"""Streamlit application for inspecting HuMID trajectory pair datasets.

Run with:
    streamlit run discriminator/streamlit_app.py -- --dataset <run_dir>

If --dataset is omitted you can pick one interactively under discriminator/datasets/.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import math

import numpy as np
import streamlit as st

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def find_run_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)


@st.cache_data(show_spinner=False)
def load_npz(path: Path):
    # Return a plain dict so Streamlit can pickle it
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@st.cache_data(show_spinner=False)
def load_pt(path: Path):
    if torch is None:
        return None
    return torch.load(path)


@st.cache_data(show_spinner=False)
def load_meta(path: Path):
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_inspection(run_dir: Path):
    insp_dir = run_dir / "inspection"
    index_path = insp_dir / "index.json"
    if not index_path.exists():
        return None
    index_data = json.loads(index_path.read_text())
    sample_path = insp_dir / index_data.get("sample_pairs", "sample_pairs.json")
    schema_path = insp_dir / index_data.get("schema", "schema.md")
    sample_pairs = json.loads(sample_path.read_text()) if sample_path.exists() else []
    schema_md = schema_path.read_text() if schema_path.exists() else ""
    return {"index": index_data, "sample_pairs": sample_pairs, "schema": schema_md}


def percentile_stats(arr: np.ndarray, pcts=(50, 75, 90, 95, 99)):
    return {f"p{p}": float(np.percentile(arr, p)) for p in pcts}


def trajectory_length_distribution(npz):
    # Reconstruct original lengths from masks (count of ones)
    m1 = npz["mask1"]
    m2 = npz["mask2"]
    lens1 = m1.sum(axis=1)
    lens2 = m2.sum(axis=1)
    return lens1, lens2


def render_stats(meta, npz):
    st.subheader("Dataset Summary")
    counts = meta["counts"]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Pairs", counts["total_pairs"])
    col_b.metric("Positives", counts["positives"])
    col_c.metric("Negatives", counts["negatives"])
    st.caption("Positives are same-expert trajectory pairs; negatives are different-expert pairs.")

    lens1, lens2 = trajectory_length_distribution(npz)
    st.write("### Padded Sequence Length")
    st.write(f"Target length (post pad/truncate): {meta['target_len']}")
    st.write("### Original Length Distribution (A side & B side)")
    tabs = st.tabs(["Histogram", "Stats"])
    with tabs[0]:
        import plotly.express as px  # Lazy import for plotting
        df = {
            "len": np.concatenate([lens1, lens2]),
            "side": ["A"] * len(lens1) + ["B"] * len(lens2)
        }
        fig = px.histogram(df, x="len", color="side", nbins=50, opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        stats_table = []
        for name, arr in [("A", lens1), ("B", lens2)]:
            row = {
                "Side": name,
                "n": int(arr.size),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
            }
            row.update(percentile_stats(arr))
            stats_table.append(row)
        st.table(stats_table)
    st.caption("Lengths are pre padding/truncation as inferred from masks.")


def render_pair_view(npz, meta, idx: int):
    st.subheader("Pair Inspection")
    label = int(npz["label"][idx])
    tag = "Matching (same expert)" if label == 1 else "Non-matching (different expert)"
    st.markdown(f"**Pair #{idx}** — {tag}")
    # For inspection, show simple summary stats; raw arrays can be large.
    a_mask = npz["mask1"][idx]
    b_mask = npz["mask2"][idx]
    a_len = int(a_mask.sum())
    b_len = int(b_mask.sum())
    col1, col2 = st.columns(2)
    col1.metric("A length", a_len)
    col2.metric("B length", b_len)

    # Optionally compute mean vectors & cosine similarity
    x1 = npz["x1"][idx][:a_len]
    x2 = npz["x2"][idx][:b_len]
    mean1 = x1.mean(axis=0)
    mean2 = x2.mean(axis=0)
    cos = float(np.dot(mean1, mean2) / ((np.linalg.norm(mean1) + 1e-9) * (np.linalg.norm(mean2) + 1e-9)))
    st.write(f"Mean feature cosine similarity: `{cos:.3f}`")

    # Dimensionality reduction (PCA 2D) for visualization
    if st.checkbox("Show PCA projection (first 2 comps)", value=False):
        from sklearn.decomposition import PCA
        max_show = st.slider("Max points per trajectory", min_value=50, max_value=meta['target_len'], value=300, step=50)
        a_trim = x1[:max_show]
        b_trim = x2[:max_show]
        concat = np.concatenate([a_trim, b_trim], axis=0)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(concat)
        a_proj = proj[:a_trim.shape[0]]
        b_proj = proj[a_trim.shape[0]:]
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_scatter(x=a_proj[:,0], y=a_proj[:,1], mode="markers", name="A", marker=dict(size=6, color="#1f77b4"))
        fig.add_scatter(x=b_proj[:,0], y=b_proj[:,1], mode="markers", name="B", marker=dict(size=6, color="#ff7f0e"))
        fig.update_layout(margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("2D PCA of raw state vectors (limited to subset).")

    if st.checkbox("Show raw arrays (truncated)", value=False):
        trunc = st.slider("Display first N timesteps", 1, min(a_len, b_len), min(25, min(a_len, b_len)))
        st.write("A[:N]")
        st.write(x1[:trunc])
        st.write("B[:N]")
        st.write(x2[:trunc])


def render_pair_side_by_side(npz, meta, dataset_idx: int, feature_idx: int):
    a_mask = npz["mask1"][dataset_idx]
    b_mask = npz["mask2"][dataset_idx]
    a_len = int(a_mask.sum())
    b_len = int(b_mask.sum())
    x1 = npz["x1"][dataset_idx][:a_len, feature_idx]
    x2 = npz["x2"][dataset_idx][:b_len, feature_idx]
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(a_len), y=x1, mode="lines+markers", name="A", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=np.arange(b_len), y=x2, mode="lines+markers", name="B", line=dict(color="#ff7f0e")))
    fig.update_layout(xaxis_title="Timestep", yaxis_title=f"Feature[{feature_idx}]", margin=dict(l=0, r=0, t=30, b=0))
    return fig, a_len, b_len


def main_streamlit(dataset_arg: str | None):
    st.title("HuMID Trajectory Pair Dataset Inspector")
    st.caption("Explore matching vs non-matching trajectory pairs and dataset statistics.")

    # Use the directory where this script lives as the base
    script_dir = Path(__file__).resolve().parent
    base = script_dir / "datasets"

    runs = find_run_dirs(base)
    if not runs:
        st.error(f"No dataset runs found under {base}.")
        return

    # base = Path("discriminator/datasets")
    # runs = find_run_dirs(base)
    # if not runs:
    #     st.error("No dataset runs found under discriminator/datasets/.")
    #     return

    dataset_path = Path(dataset_arg) if dataset_arg else None
    if dataset_path is None:
        chosen = st.selectbox("Select run directory", runs, format_func=lambda p: p.name)
    else:
        chosen = dataset_path
        st.info(f"Using dataset provided via CLI: {chosen}")

    meta_path = chosen / "meta.json"
    npz_path = chosen / "pairs.npz"
    if not meta_path.exists() or not npz_path.exists():
        st.error("Selected run directory missing required files (meta.json / pairs.npz)")
        return

    meta = load_meta(meta_path)
    npz = load_npz(npz_path)
    inspection = load_inspection(chosen)

    # Sidebar controls
    st.sidebar.header("Navigation")
    total = meta['counts']['total_pairs']
    label_filter = st.sidebar.selectbox("Label filter", ["All", "Positives", "Negatives"])
    indices = np.arange(total)
    if label_filter != "All":
        lab = 1 if label_filter == "Positives" else 0
        indices = indices[npz['label'][indices] == lab]
    idx = st.sidebar.number_input("Pair index", min_value=0, max_value=int(indices.size - 1), value=0, step=1)
    actual_idx = int(indices[idx]) if indices.size else 0

    st.sidebar.write(f"Filtered count: {indices.size}")
    if st.sidebar.button("Random pair"):
        import random
        actual_idx = int(random.choice(indices)) if indices.size else 0
    st.sidebar.write(f"Viewing actual pair id: {actual_idx}")

    tabs = st.tabs(["Overview", "Pair", "Inspection", "Report.md"])
    with tabs[0]:
        render_stats(meta, npz)
    with tabs[1]:
        render_pair_view(npz, meta, actual_idx)
    with tabs[2]:
        if inspection is None:
            st.info("No inspection folder found in this run.")
        else:
            st.subheader("Inspection Files")
            st.caption("Loaded from inspection/index.json, sample_pairs.json, schema.md")
            st.json(inspection["index"])
            if inspection.get("schema"):
                with st.expander("Schema (schema.md)", expanded=False):
                    st.markdown(inspection["schema"])
            sample_pairs = inspection.get("sample_pairs", [])
            st.write(f"Sample pairs available: {len(sample_pairs)}")
            if not sample_pairs:
                st.warning("No sample_pairs.json found or it is empty.")
            else:
                label_filter = st.selectbox("Sample label filter", ["All", "Positives", "Negatives"], key="insp_label_filter")
                filtered = [p for p in sample_pairs if label_filter == "All" or (p["label"] == 1 if label_filter == "Positives" else p["label"] == 0)]
                if not filtered:
                    st.warning("No samples match this filter.")
                else:
                    choices = [f"#{i} | global {p['dataset_index']} | label {p['label']} | A:{p['a_expert']} B:{p['b_expert']}" for i, p in enumerate(filtered)]
                    sel_idx = st.selectbox("Select sample", list(range(len(filtered))), format_func=lambda i: choices[i])
                    selected = filtered[int(sel_idx)]
                    st.write({k: selected[k] for k in selected})
                    ds_idx = int(selected["dataset_index"])
                    feat_max = int(meta["state_dim"]) - 1
                    feature_idx = st.slider("Feature index for side-by-side plot", 0, feat_max, min(1, feat_max))
                    fig, a_len, b_len = render_pair_side_by_side(npz, meta, ds_idx, feature_idx)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Pair lengths — A: {a_len}, B: {b_len}; label: {selected['label']} (1=matching, 0=non-matching).")
    with tabs[3]:
        report_path = chosen / "report.md"
        if report_path.exists():
            st.markdown(report_path.read_text())
        else:
            st.write("No report.md present.")


def parse_cli_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dataset", type=str, default=None, help="Specific run directory to load")
    # Streamlit passes unknown args; ignore others
    args, _ = ap.parse_known_args()
    return args


if __name__ == "__main__":
    cli_args = parse_cli_args()
    main_streamlit(cli_args.dataset)
