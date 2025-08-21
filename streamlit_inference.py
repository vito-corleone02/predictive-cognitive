import os
import glob
import pickle
import random
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Optional Streamlit import for GUI mode only
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None



# Explainability
import shap
from lime.lime_tabular import LimeTabularExplainer



# =============================================================================
# 0) CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PREDICT_DATA_DIR", os.path.join(BASE_DIR, "data"))



# Which modalities to use (will load if file exists)
MODALITIES = [
    "audio",
    "fkps",
    "gaze_conf",
    "pose_conf",
    "text",
    # New image modalities (optional, only used if features are prepared)
    "mri",
    "pet",
]



# =============================================================================
# File pattern helpers
# =============================================================================
def candidate_paths(split: str, pid: int, modality: str) -> List[str]:
    paths: List[str] = []
    if split == "train":
        base_dirs = [os.path.join(DATA_DIR, "train")]
        prefixes = ["train"]
    elif split == "valid":
        base_dirs = [os.path.join(DATA_DIR, "valid"), os.path.join(DATA_DIR, "dev")]
        prefixes = ["dev", "valid"]
    elif split == "test":
        base_dirs = [os.path.join(DATA_DIR, "test")]
        prefixes = ["test"]
    else:
        base_dirs = [os.path.join(DATA_DIR, split)]
        prefixes = [split]

    for base in base_dirs:
        for pref in prefixes:
            flat = os.path.join(base, f"{pref}_ft_{modality}_{pid}.npy")
            nested = os.path.join(base, str(pid), f"{pref}_ft_{modality}_{pid}.npy")
            paths.append(flat)
            paths.append(nested)

    # dedupe preserve order
    seen = set()
    unique_paths: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths

def first_existing(path_list: List[str]) -> Optional[str]:
    for p in path_list:
        if os.path.exists(p):
            return p
    return None









# =============================================================================
# 2) FEATURE LOADING (Same approach as demo_inference.py)
# =============================================================================
def load_features_for_id(split: str, pid: int) -> Tuple[np.ndarray, List[str]]:
    """Load & align features for one participant using the same approach as demo_inference.py."""
    feats, names = [], []
    for modality in MODALITIES:
        path = first_existing(candidate_paths(split, pid, modality))
        if path is None:
            continue
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr)
        # Use the same summarization as demo_inference.py
        arr = np.nan_to_num(arr.astype(np.float32))
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        feat_vec = arr
        feat_names_mod = [f"{modality}_{i}" for i in range(arr.shape[0])]
        feats.append(feat_vec.astype(np.float32, copy=False))
        names.extend(feat_names_mod)

    if len(feats) == 0:
        return np.array([]), []
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False), names

def load_features_for_pid(split: str, pid: int) -> np.ndarray:
    """Load & align features for one participant - same as demo_inference.py."""
    feats, names = load_features_for_id(split, pid)
    if feats.size == 0:
        raise RuntimeError(f"No features found for PID={pid}")
    return feats

def predict_for_ids(split: str, ids: List[int], model, scaler) -> pd.DataFrame:
    """Run inference on a list of participant IDs - same as demo_inference.py.
    
    Usage example (same as demo_inference.py):
        participant_ids = [300, 301]
        results = predict_for_ids("test", participant_ids, model, scaler)
        print(results)
        results.to_csv("artifacts/inference_results.csv", index=False)
    """
    X_rows = []
    valid_ids = []
    for pid in ids:
        try:
            vec = load_features_for_pid(split, pid)
            X_rows.append(vec)
            valid_ids.append(pid)
        except RuntimeError as e:
            print(f"Skipping {pid}: {e}")
            continue
    if not X_rows:
        raise RuntimeError("No valid participants found for inference.")
    X = np.vstack(X_rows)
    X_s = scaler.transform(X)
    preds = model.predict(X_s)
    return pd.DataFrame({"Participant_ID": valid_ids, "Predicted_PHQ8": preds})



def check_pretrained_model() -> bool:
    """Check if pretrained model exists and is valid."""
    model_path = os.path.join("artifacts", "severity_model.pkl")
    if not os.path.exists(model_path):
        return False
    try:
        with open(model_path, "rb") as f:
            model_artifact = pickle.load(f)
        required_keys = ["model", "scaler", "feature_names"]
        return all(key in model_artifact for key in required_keys)
    except Exception:
        return False

def run_simple_inference() -> Dict[str, Any]:
    """Run simple inference using pretrained model."""
    # Load pretrained model
    model_path = os.path.join("artifacts", "severity_model.pkl")
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)
    
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feat_names = model_artifact["feature_names"]
    
    # For demo purposes, create some sample data
    # In real usage, you would load actual test data
    n_samples = 100
    X_sample = np.random.randn(n_samples, len(feat_names)).astype(np.float32)
    X_sample_s = scaler.transform(X_sample)
    pred_sample = model.predict(X_sample_s)
    
    # Create explainability objects
    explainer_shap = shap.TreeExplainer(model)
    explainer_lime = LimeTabularExplainer(
        X_sample_s,
        feature_names=feat_names,
        class_names=["PHQ8_Score"],
        verbose=False,
        mode="regression"
    )
    
    return {
        "model": model,
        "scaler": scaler,
        "feat_names": feat_names,
        "X_sample_s": X_sample_s,
        "pred_sample": pred_sample,
        "TOTAL_N": n_samples,
        "explainer_shap": explainer_shap,
        "explainer_lime": explainer_lime,
    }



# =============================================================================
# 4) PHYSIOLOGICAL MARKERS SIMULATION
# =============================================================================
def simulate_physiological_markers(n_samples, breathing_range=(12, 20), tapping_range=(1, 5), heart_rate_range=(60, 100)):
    """
    Simulate physiological markers with customizable ranges.
    Args:
        n_samples: Number of samples to generate
        breathing_range: Tuple of (min, max) breathing rate in breaths per minute
        tapping_range: Tuple of (min, max) tapping rate in taps per second  
        heart_rate_range: Tuple of (min, max) heart rate in beats per minute
    Returns:
        Array of shape (n_samples, 3) with [breathing, tapping, heart_rate]
    """
    breathing = np.random.uniform(breathing_range[0], breathing_range[1], size=(n_samples, 1))
    tapping = np.random.uniform(tapping_range[0], tapping_range[1], size=(n_samples, 1))
    heart_rate = np.random.uniform(heart_rate_range[0], heart_rate_range[1], size=(n_samples, 1))
    return np.hstack([breathing, tapping, heart_rate])

# =============================================================================
# 5) MISINFORMATION SIMULATION
# =============================================================================
def simulate_misinformation(num_nodes, init_infected_frac=0.1, trans_prob=0.2, rec_prob=0.1, steps=20):
    G = nx.barabasi_albert_graph(num_nodes, m=2, seed=42)
    for n in G.nodes():
        G.nodes[n]['state'] = 'S'
    infected = random.sample(list(G.nodes()), max(1, int(init_infected_frac * num_nodes)))
    for n in infected:
        G.nodes[n]['state'] = 'I'
    S_list, I_list, R_list = [], [], []
    for _ in range(steps):
        new_states = {}
        for n in G.nodes():
            state = G.nodes[n]['state']
            if state == 'S':
                for nbr in G.neighbors(n):
                    if G.nodes[nbr]['state'] == 'I' and random.random() < trans_prob:
                        new_states[n] = 'I'
                        break
            elif state == 'I':
                if random.random() < rec_prob:
                    new_states[n] = 'R'
        for n, s in new_states.items():
            G.nodes[n]['state'] = s
        states = [G.nodes[n]['state'] for n in G.nodes()]
        S_list.append(states.count('S'))
        I_list.append(states.count('I'))
        R_list.append(states.count('R'))
    return S_list, I_list, R_list, G

def allocate_resources(severity_scores, capacity=10):
    idx = np.argsort(severity_scores)[::-1]
    return idx[:capacity], idx[capacity:]

# =============================================================================
# 5) STREAMLIT APP (NEW UI)
# =============================================================================
def run_app():
    # ---- Header ----
    st.title("🧠 Resource Allocation Using Multimodal AI & Misinformation Modeling in Healthcare")
    st.caption("The UI below balances multimodal data interaction (audio, image, physiological signals) and real-time simulation of misinformation spread to prioritize patients for limited care resources.")



    # Model status check
    if not check_pretrained_model():
        st.sidebar.error("❌ Pretrained model not found or invalid!")
        st.sidebar.info("Please ensure 'artifacts/severity_model.pkl' exists and contains a valid model.")
    else:
        st.sidebar.success("✅ Pretrained model ready!")
    
    # Sidebar controls (misinfo + capacity)
    st.sidebar.header("Simulation & Allocation Controls")
    trans_prob = st.sidebar.slider("Transmission Probability", 0.0, 1.0, 0.2, 0.01)
    rec_prob   = st.sidebar.slider("Recovery Probability", 0.0, 1.0, 0.1, 0.01)
    steps      = st.sidebar.slider("Steps", 5, 100, 20, 1)
    capacity   = st.sidebar.number_input("Treatment Capacity", min_value=1, max_value=500, value=10)
    method     = st.sidebar.radio("Explanation Method", ["LIME", "SHAP"], index=0, horizontal=True)
    
    # Physiological markers controls
    st.sidebar.header("🧬 Physiological Markers")
    breathing_min = st.sidebar.number_input("Breathing Min (bpm)", min_value=8, max_value=30, value=12, step=1)
    breathing_max = st.sidebar.number_input("Breathing Max (bpm)", min_value=8, max_value=30, value=20, step=1)
    tapping_min = st.sidebar.number_input("Tapping Min (taps/sec)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
    tapping_max = st.sidebar.number_input("Tapping Max (taps/sec)", min_value=0.5, max_value=10.0, value=5.0, step=0.5)
    heart_rate_min = st.sidebar.number_input("Heart Rate Min (bpm)", min_value=40, max_value=200, value=60, step=5)
    heart_rate_max = st.sidebar.number_input("Heart Rate Max (bpm)", min_value=40, max_value=200, value=100, step=5)
    

    # Uploaders (optional)
    st.subheader("📥 Upload Audio & Image (Optional)")
    up_col1, up_col2, up_col3 = st.columns([1,1,1])
    with up_col1:
        st.file_uploader("Upload Audio Files", type=["wav", "mp3", "flac"], accept_multiple_files=True, key="audio_uploads")
    with up_col2:
        st.file_uploader("Upload Image Files", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True, key="image_uploads")
    with up_col3:
        st.number_input("n Samples", min_value=1, value=10, step=1, key="n_samples_ui")

    # Run inference button
    if check_pretrained_model():
        if st.button("▶️ Run Inference"):
            with st.spinner("Loading pretrained model and running inference..."):
                arts = run_simple_inference()
                st.session_state["arts"] = arts
    else:
        st.button("▶️ Run Inference", disabled=True)
        st.warning("Cannot run inference: Pretrained model not available.")
            
    # Physiological markers simulation
    if st.button("🧬 Simulate Physiological Data"):
        n_samples = st.session_state.get("n_samples_ui", 10)
        physio_data = simulate_physiological_markers(
            n_samples=n_samples,
            breathing_range=(breathing_min, breathing_max),
            tapping_range=(tapping_min, tapping_max),
            heart_rate_range=(heart_rate_min, heart_rate_max)
        )
        st.session_state["physio_data"] = physio_data
        st.session_state["show_physio"] = True
        st.success(f"Generated {n_samples} physiological samples!")


    if "arts" not in st.session_state:
        st.info("Click **Run Inference** to load the pretrained model and run inference.")
        return

    arts = st.session_state["arts"]

    # Dataset Summary
    st.subheader("📊 Inference Results")
    st.write(
        f"**Samples**: {arts['TOTAL_N']}  |  **Features**: {len(arts['feat_names'])}"
    )

    # Misinformation Simulation for TOTAL_N
    S_list_, I_list_, R_list_, G_net_ = simulate_misinformation(
        num_nodes=arts["TOTAL_N"], trans_prob=trans_prob, rec_prob=rec_prob, steps=steps
    )
    misinfo_risk_ = I_list_[-1] / arts["TOTAL_N"]

    # Adjusted severities + allocation
    adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_)
    treated, untreated = allocate_resources(adjusted_all_, capacity=capacity)

    # Patient table (first 100 for speed)
    df_all = pd.DataFrame({
        "Patient ID": list(range(len(adjusted_all_))),
        "Raw Severity": np.round(arts["pred_sample"], 3),
        "Adjusted Severity": np.round(adjusted_all_, 3),
        "Priority": ["✅ Yes" if i in treated else "❌ No" for i in range(len(adjusted_all_))]
    })
    st.dataframe(df_all.head(100), use_container_width=True)

    # Patient Details & Explanations
    st.subheader("📊 Patient Details and Explanations")
    patient_idx = st.selectbox("Select Patient ID:", options=list(range(len(adjusted_all_))), index=0)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Raw Severity", f"{arts['pred_sample'][patient_idx]:.2f}")
    with col2:
        st.metric("Adjusted Severity", f"{adjusted_all_[patient_idx]:.2f}")
    with col3:
        st.metric("Prioritized for Treatment", "Yes" if patient_idx in treated else "No")
    with col4:
        st.metric("Misinformation Risk", f"{misinfo_risk_:.2f}")

    # Explanation block (LIME default like your sketch; SHAP optional)
    if method == "LIME":
        st.subheader("🔍 LIME Explanation")
        lime_exp = arts["explainer_lime"].explain_instance(
            arts["X_sample_s"][patient_idx],
            arts["model"].predict,
            num_features=min(10, len(arts["feat_names"]))
        )
        fig = lime_exp.as_pyplot_figure()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.subheader("🧠 SHAP Explanation")
        shap_vals_local = arts["explainer_shap"].shap_values(arts["X_sample_s"][patient_idx:patient_idx+1])
        shap.force_plot(
            arts["explainer_shap"].expected_value,
            shap_vals_local[0],
            features=arts["X_sample_s"][patient_idx:patient_idx+1],
            matplotlib=True, show=False
        )
        fig_local = plt.gcf()
        st.pyplot(fig_local, use_container_width=True)
        plt.close(fig_local)



    # Misinformation Spread Over Time
    st.subheader("📉 Misinformation Spread Over Time")
    fig_misinfo, ax_misinfo = plt.subplots()
    ax_misinfo.plot(S_list_, label="Susceptible")
    ax_misinfo.plot(I_list_, label="Infected")
    ax_misinfo.plot(R_list_, label="Recovered")
    ax_misinfo.legend()
    ax_misinfo.set_xlabel("Step")
    ax_misinfo.set_ylabel("Nodes")
    st.pyplot(fig_misinfo, use_container_width=True)
    plt.close(fig_misinfo)

    # Network Snapshot
    st.subheader("🌐 Final Network State (Social Network Visualization)")
    fig_net, ax_net = plt.subplots(figsize=(7, 5))
    pos = nx.spring_layout(G_net_, seed=42)
    c_map = {'S': 'blue', 'I': 'red', 'R': 'green'}
    node_colors = [c_map[G_net_.nodes[n]['state']] for n in G_net_.nodes()]
    nx.draw(G_net_, pos, node_color=node_colors, node_size=20, with_labels=False, ax=ax_net)
    st.pyplot(fig_net, use_container_width=True)
    plt.close(fig_net)


    




if __name__ == "__main__":
    # If running under Streamlit
    if st is not None and (os.environ.get("STREAMLIT_SERVER_PORT") or "streamlit" in os.path.basename(sys.argv[0]).lower()):
        run_app()
    else:
        parser = argparse.ArgumentParser(description="PHQ8 inference using pretrained model (CLI or Streamlit)")
        parser.add_argument("--mode", choices=["cli", "app"], default="app")
        parser.add_argument("--trans-prob", type=float, default=0.2)
        parser.add_argument("--rec-prob", type=float, default=0.1)
        parser.add_argument("--steps", type=int, default=20)
        parser.add_argument("--capacity", type=int, default=10)
        parser.add_argument("--breathing-min", type=float, default=12.0, help="Minimum breathing rate (bpm)")
        parser.add_argument("--breathing-max", type=float, default=20.0, help="Maximum breathing rate (bpm)")
        parser.add_argument("--tapping-min", type=float, default=1.0, help="Minimum tapping rate (taps/sec)")
        parser.add_argument("--tapping-max", type=float, default=5.0, help="Maximum tapping rate (taps/sec)")
        parser.add_argument("--heart-rate-min", type=float, default=60.0, help="Minimum heart rate (bpm)")
        parser.add_argument("--heart-rate-max", type=float, default=100.0, help="Maximum heart rate (bpm)")
        parser.add_argument("--physio-samples", type=int, default=10, help="Number of physiological samples to generate")

        args = parser.parse_args()

        if args.mode == "app":
            if st is None:
                print("Streamlit not available. Install it or run with --mode cli.")
                sys.exit(1)
            run_app()
            sys.exit(0)



        # Simple inference example (same as demo_inference.py)
        print("\n=== SIMPLE INFERENCE EXAMPLE ===")
        try:
            # Example: predict for participants 300 and 301 in the "test" split
            participant_ids = [300, 301]
            # Load model first for inference
            model_path = os.path.join("artifacts", "severity_model.pkl")
            with open(model_path, "rb") as f:
                model_artifact = pickle.load(f)
            model = model_artifact["model"]
            scaler = model_artifact["scaler"]
            
            results = predict_for_ids("test", participant_ids, model, scaler)
            print("Inference results:")
            print(results)
            results.to_csv("artifacts/inference_results.csv", index=False)
            print("Saved predictions → artifacts/inference_results.csv")
        except Exception as e:
            print(f"Inference example failed: {e}")
        
        print("\nRunning simple inference in CLI mode...\n")
        
        # Check if pretrained model exists
        if not check_pretrained_model():
            print("❌ Error: Pretrained model not found or invalid!")
            print("Please ensure 'artifacts/severity_model.pkl' exists and contains a valid model.")
            sys.exit(1)
        
        # Run simple inference
        arts = run_simple_inference()
        TOTAL_N = arts["TOTAL_N"]

        S_list_, I_list_, R_list_, G_net_ = simulate_misinformation(
            num_nodes=TOTAL_N, trans_prob=args.trans_prob, rec_prob=args.rec_prob, steps=args.steps,
        )
        misinfo_risk_ = I_list_[-1] / TOTAL_N
        adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_)
        treated, untreated = allocate_resources(adjusted_all_, capacity=args.capacity)

        print(f"Misinformation risk: {misinfo_risk_:.3f}")
        print(f"Treatment capacity: {args.capacity}")
        print(f"Top {min(len(treated), args.capacity)} prioritized indices (global): {treated.tolist()}")
        
        # Generate and display physiological markers
        print(f"\n=== PHYSIOLOGICAL MARKERS SIMULATION ===")
        physio_data = simulate_physiological_markers(
            n_samples=args.physio_samples,
            breathing_range=(args.breathing_min, args.breathing_max),
            tapping_range=(args.tapping_min, args.tapping_max),
            heart_rate_range=(args.heart_rate_min, args.heart_rate_max)
        )
        print(f"Generated {args.physio_samples} physiological samples:")
        print(f"Breathing range: {args.breathing_min}-{args.breathing_max} bpm")
        print(f"Tapping range: {args.tapping_min}-{args.tapping_max} taps/sec")
        print(f"Heart rate range: {args.heart_rate_min}-{args.heart_rate_max} bpm")
        print(f"Sample means - Breathing: {np.mean(physio_data[:, 0]):.2f}, Tapping: {np.mean(physio_data[:, 1]):.2f}, HR: {np.mean(physio_data[:, 2]):.2f}")

        # Save heatmap
        os.makedirs("artifacts", exist_ok=True)
        fig, ax = plt.subplots()
        scatter = ax.scatter(range(len(adjusted_all_)), np.zeros(len(adjusted_all_)), c=adjusted_all_, cmap="Reds", s=40)
        plt.colorbar(scatter, label="Adjusted PHQ8")
        ax.set_yticks([])
        ax.set_xlabel("Global Index")
        ax.set_title("Adjusted Risk Heatmap (CLI)")
        plt.tight_layout()
        heatmap_path = os.path.join("artifacts", "risk_heatmap_cli.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved heatmap: {heatmap_path}")
