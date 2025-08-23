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
from PIL import Image

# Conditional OpenCV import for cloud compatibility
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. Using PIL for image processing.")

# Optional Streamlit import for GUI mode only
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

# Audio processing
try:
    import librosa
except ImportError:
    librosa = None
    print("Warning: librosa not available. Audio processing will be disabled.")

# Explainability
import shap
from lime.lime_tabular import LimeTabularExplainer

# Machine learning utilities
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None
    print("Warning: sklearn not available. Some audio analysis features will be disabled.")



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
]



# =============================================================================
# ALZHEIMER'S DISEASE IMAGE PROCESSING
# =============================================================================
def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess an image for Alzheimer's disease classification."""
    try:
        # Load image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            if OPENCV_AVAILABLE:
                img = cv2.imread(image_path)
                if img is None:
                    # Try PIL as fallback
                    pil_img = Image.open(image_path)
                    img = np.array(pil_img)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                # PIL fallback
                pil_img = Image.open(image_path)
                img = np.array(pil_img)
                # Convert RGB to BGR if needed (OpenCV format)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = img[:, :, ::-1]  # RGB to BGR
                elif len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)  # Grayscale to 3-channel
        else:
            # Handle .PNG files (case sensitive)
            if OPENCV_AVAILABLE:
                img = cv2.imread(image_path)
                if img is None:
                    return None
            else:
                # PIL fallback
                try:
                    pil_img = Image.open(image_path)
                    img = np.array(pil_img)
                    # Convert RGB to BGR if needed (OpenCV format)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = img[:, :, ::-1]  # RGB to BGR
                    elif len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=-1)  # Grayscale to 3-channel
                except Exception:
                    return None

        if img is None:
            return None

        # Resize
        if OPENCV_AVAILABLE:
            img = cv2.resize(img, target_size)
        else:
            # PIL resize
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
            img = np.array(pil_img)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to grayscale if it's a 3-channel image (for MRI/PET)
        if len(img.shape) == 3:
            if OPENCV_AVAILABLE:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                # PIL grayscale conversion
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                pil_img = pil_img.convert('L')
                img = np.array(pil_img) / 255.0
            img = np.expand_dims(img, axis=-1)

        return img

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def extract_image_features(img: np.ndarray, target_features: int = 50436) -> np.ndarray:
    """Extract features from preprocessed image with padding/truncation to match target dimensions."""
    try:
        # Flatten the image
        flattened = img.flatten()
        
        # Basic statistical features
        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        # Histogram features (simplified)
        hist, _ = np.histogram(img.flatten(), bins=10, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        
        # Combine all features
        all_features = np.concatenate([
            flattened,
            [mean_val, std_val, min_val, max_val],
            hist
        ])
        
        # Pad or truncate to match expected feature dimensions
        if all_features.size < target_features:
            # Pad with zeros if we have fewer features
            padding = np.zeros(target_features - all_features.size, dtype=np.float32)
            all_features = np.concatenate([all_features, padding])
        elif all_features.size > target_features:
            # Truncate if we have more features
            all_features = all_features[:target_features]
        
        return all_features.astype(np.float32)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.array([])

def check_alzheimer_model() -> bool:
    """Check if Alzheimer model exists and is valid."""
    model_path = os.path.join("artifacts", "alzheimer_classifier.pkl")
    if not os.path.exists(model_path):
        return False
    try:
        with open(model_path, "rb") as f:
            model_artifact = pickle.load(f)
        required_keys = ["classifier", "scaler", "label_encoder", "classes"]
        if not all(key in model_artifact for key in required_keys):
            return False
        
        # Check if scaler has the expected feature count
        scaler = model_artifact["scaler"]
        if hasattr(scaler, 'n_features_in_'):
            print(f"Alzheimer model expects {scaler.n_features_in_} features")
        
        return True
    except Exception:
        return False

def load_alzheimer_model() -> Dict[str, Any]:
    """Load the trained Alzheimer classifier model."""
    model_path = os.path.join("artifacts", "alzheimer_classifier.pkl")
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)
    return model_artifact

def classify_alzheimer_image(image_file, alzheimer_model: Dict[str, Any]) -> Tuple[str, float, np.ndarray]:
    """Classify an uploaded image using the Alzheimer model."""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_upload_{random.randint(1000, 9999)}.png"
        with open(temp_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        # Load and preprocess image
        img = load_and_preprocess_image(temp_path)
        if img is None:
            os.remove(temp_path)
            return "Error: Could not load image", 0.0, np.array([])
        
        # Extract features with proper dimension matching
        target_features = alzheimer_model["scaler"].n_features_in_
        features = extract_image_features(img, target_features=target_features)
        if features.size == 0:
            os.remove(temp_path)
            return "Error: Could not extract features", 0.0, np.array([])
        
        # Scale features
        scaler = alzheimer_model["scaler"]
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict
        classifier = alzheimer_model["classifier"]
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        
        # Get class name and confidence
        label_encoder = alzheimer_model["label_encoder"]
        class_name = label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return class_name, confidence, img
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0, np.array([])



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

# --- Audio MFCC features ---
def extract_mfcc_features(audio_files):
    """Extract MFCC features from audio files."""
    if librosa is None:
        return None, "librosa not available for audio processing"
    
    try:
        mfcc_feats = []
        file_names = []
        
        for file in audio_files:
            # Save uploaded file temporarily
            temp_path = f"temp_audio_{random.randint(1000, 9999)}.wav"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                y, sr = librosa.load(temp_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_feats.append(mfcc_mean)
                file_names.append(file.name)
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                print(f"Error processing audio file {file.name}: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                continue
        
        if not mfcc_feats:
            return None, "No valid audio files could be processed"
        
        return np.array(mfcc_feats), file_names
        
    except Exception as e:
        return None, f"Error in MFCC extraction: {str(e)}"

def analyze_audio_features(mfcc_features, file_names):
    """Analyze extracted MFCC features and create results similar to inference results."""
    try:
        if mfcc_features is None or mfcc_features.size == 0:
            return None
        
        # Calculate basic statistics for each feature
        feature_stats = {
            'mean': np.mean(mfcc_features, axis=0),
            'std': np.std(mfcc_features, axis=0),
            'min': np.min(mfcc_features, axis=0),
            'max': np.max(mfcc_features, axis=0)
        }
        
        # Create feature names for MFCC coefficients
        mfcc_feature_names = [f"MFCC_{i+1}" for i in range(mfcc_features.shape[1])]
        
        # Calculate similarity matrix between audio files
        if cosine_similarity is None:
            print("Warning: sklearn not available for cosine similarity. Skipping audio similarity matrix.")
            similarity_matrix = None
        else:
            similarity_matrix = cosine_similarity(mfcc_features)
        
        # Create a summary dataframe
        summary_df = pd.DataFrame({
            'File Name': file_names,
            'MFCC_Mean': [np.mean(feat) for feat in mfcc_features],
            'MFCC_Std': [np.std(feat) for feat in mfcc_features],
            'MFCC_Range': [np.max(feat) - np.min(feat) for feat in mfcc_features]
        })
        
        return {
            'mfcc_features': mfcc_features,
            'file_names': file_names,
            'feature_names': mfcc_feature_names,
            'feature_stats': feature_stats,
            'similarity_matrix': similarity_matrix,
            'summary_df': summary_df,
            'total_files': len(file_names),
            'mfcc_dimensions': mfcc_features.shape[1]
        }
        
    except Exception as e:
        return None

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
    
    # Alzheimer model status check
    if not check_alzheimer_model():
        st.sidebar.error("❌ Alzheimer model not found or invalid!")
        st.sidebar.info("Please ensure 'artifacts/alzheimer_classifier.pkl' exists and contains a valid model.")
    else:
        st.sidebar.success("✅ Alzheimer model ready!")
    
    # OpenCV status check
    if OPENCV_AVAILABLE:
        st.sidebar.success("✅ OpenCV ready!")
    else:
        st.sidebar.warning("⚠️ OpenCV not available!")
        st.sidebar.info("Using PIL fallback for image processing. Some advanced image features may be limited.")
    
    # Audio processing status check
    if librosa is None:
        st.sidebar.warning("⚠️ Audio processing disabled!")
        st.sidebar.info("Install librosa for audio file analysis: pip install librosa")
    else:
        st.sidebar.success("✅ Audio processing ready!")
    
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
    st.markdown(
        """
        <style>
        /* Tighten inner cell padding and add header row vibe */
        .cell-header { font-weight: 700; border-bottom: 1px solid #bbb; margin: -0.25rem -0.25rem 0.5rem -0.25rem; padding: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        up_col1, up_col2, up_col3 = st.columns([1,1,1])
        with up_col1:
            with st.container(border=True):
                st.markdown('<div class="cell-header">Upload Audio Files</div>', unsafe_allow_html=True)
                audio_files = st.file_uploader(
                    "Upload Audio Files",
                    type=["wav", "mp3", "flac"],
                    accept_multiple_files=True,
                    key="audio_uploads",
                    label_visibility="collapsed",
                )
                # Add audio processing button
                if audio_files and len(audio_files) > 0:
                    if st.button("🔊 Process Audio Files", key="process_audio"):
                        with st.spinner("Processing audio files..."):
                            if librosa is None:
                                st.error("❌ librosa not available. Please install it for audio processing.")
                            else:
                                # Extract MFCC features
                                mfcc_features, file_names = extract_mfcc_features(audio_files)
                                
                                if mfcc_features is not None:
                                    # Analyze features
                                    audio_results = analyze_audio_features(mfcc_features, file_names)
                                    if audio_results:
                                        st.session_state["audio_results"] = audio_results
                                        st.success(f"✅ Processed {len(audio_files)} audio files!")
                                    else:
                                        st.error("❌ Failed to analyze audio features")
                                else:
                                    st.error(f"❌ Audio processing failed: {file_names}")
                else:
                    # Clear audio results when no audio files are uploaded
                    if "audio_results" in st.session_state:
                        del st.session_state["audio_results"]
        with up_col2:
            with st.container(border=True):
                st.markdown('<div class="cell-header">Upload Image Files</div>', unsafe_allow_html=True)
                st.file_uploader(
                    "Upload Image Files",
                    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                    accept_multiple_files=True,
                    key="image_uploads",
                    label_visibility="collapsed",
                )
        with up_col3:
            with st.container(border=True):
                st.markdown('<div class="cell-header">n Samples ▼</div>', unsafe_allow_html=True)
                st.number_input(
                    "n Samples",
                    min_value=1,
                    value=10,
                    step=1,
                    key="n_samples_ui",
                    label_visibility="collapsed",
                )


    # Alzheimer Image Classification
    if check_alzheimer_model():
        st.subheader("🧠 Alzheimer's Disease Image Classification")
        
        # Show OpenCV status for image processing
        if not OPENCV_AVAILABLE:
            st.info("ℹ️ **Note:** OpenCV is not available. Using PIL for image processing. This may affect some advanced image features.")
        
        # Load Alzheimer model
        if "alzheimer_model" not in st.session_state:
            with st.spinner("Loading Alzheimer model..."):
                alzheimer_model = load_alzheimer_model()
                st.session_state["alzheimer_model"] = alzheimer_model
        
        # Process uploaded images
        if "image_uploads" in st.session_state and st.session_state["image_uploads"]:
            st.write("**Processing uploaded images...**")
            
            for i, uploaded_file in enumerate(st.session_state["image_uploads"]):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**Image {i+1}:** {uploaded_file.name}")
                        # Display the uploaded image
                        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                    
                    with col2:
                        if st.button(f"🔍 Classify Image {i+1}", key=f"classify_{i}"):
                            with st.spinner("Classifying image..."):
                                alzheimer_model = st.session_state["alzheimer_model"]
                                prediction, confidence, processed_img = classify_alzheimer_image(uploaded_file, alzheimer_model)
                                
                                if prediction.startswith("Error"):
                                    st.error(prediction)
                                else:
                                    # Display results - only prediction
                                    st.success(f"**Prediction:** {prediction}")
        else:
            st.info("Upload images above to classify them for Alzheimer's disease.")
    else:
        st.warning("Alzheimer model not available. Cannot perform image classification.")
    
    # Audio Analysis Results
    if "audio_results" in st.session_state:
        st.subheader("🎵 Audio Analysis Results")
        audio_results = st.session_state["audio_results"]
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", audio_results["total_files"])
        with col2:
            st.metric("MFCC Dimensions", audio_results["mfcc_dimensions"])
        with col3:
            st.metric("Features per File", len(audio_results["feature_names"]))
        with col4:
            avg_mfcc = np.mean(audio_results["mfcc_features"])
            st.metric("Avg MFCC Value", f"{avg_mfcc:.3f}")
        
        # Display summary table
        st.write("**Audio Files Summary:**")
        st.dataframe(audio_results["summary_df"], use_container_width=True)
    
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
        
        # Apply custom color scheme to LIME chart
        custom_colors = ['#003A6B', '#1B5886', '#3776A1', '#5293BB', '#6EB1D6', '#89CFF1']
        
        # Get the axes and modify colors
        ax = fig.gca()
        bars = ax.patches
        
        # Apply custom colors to bars (cycling through the palette)
        for i, bar in enumerate(bars):
            color_idx = i % len(custom_colors)
            bar.set_color(custom_colors[color_idx])
            bar.set_alpha(0.8)  # Add some transparency for better aesthetics
        
        # Update the figure style
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')  # Light background
        
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
    ax_misinfo.plot(S_list_, label="Susceptible", color='#003A6B', linewidth=2)
    ax_misinfo.plot(I_list_, label="Infected", color='#3776A1', linewidth=2)
    ax_misinfo.plot(R_list_, label="Recovered", color='#89CFF1', linewidth=2)
    ax_misinfo.legend()
    ax_misinfo.set_xlabel("Step")
    ax_misinfo.set_ylabel("Nodes")
    st.pyplot(fig_misinfo, use_container_width=True)
    plt.close(fig_misinfo)

    # Network Snapshot
    st.subheader("🌐 Final Network State (Social Network Visualization)")
    fig_net, ax_net = plt.subplots(figsize=(7, 5))
    pos = nx.spring_layout(G_net_, seed=42)
    c_map = {'S': '#003A6B', 'I': '#1B5886', 'R': '#3776A1'}
    node_colors = [c_map[G_net_.nodes[n]['state']] for n in G_net_.nodes()]
    nx.draw(G_net_, pos, node_color=node_colors, node_size=20, with_labels=False, ax=ax_net, edge_color='gray')
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
        parser.add_argument("--alzheimer-image", type=str, help="Path to image file for Alzheimer classification")
        parser.add_argument("--audio-files", nargs="+", help="Paths to audio files for MFCC analysis")

        args = parser.parse_args()

        if args.mode == "app":
            if st is None:
                print("Streamlit not available. Install it or run with --mode cli.")
                sys.exit(1)
            run_app()
            sys.exit(0)



        # Simple inference example (same as demo_inference.py)
        print("\n=== SIMPLE INFERENCE EXAMPLE ===")
        print("Usage examples:")
        print("  - Basic inference: python streamlit_inference.py --mode cli")
        print("  - With audio files: python streamlit_inference.py --mode cli --audio-files file1.wav file2.mp3")
        print("  - With Alzheimer image: python streamlit_inference.py --mode cli --alzheimer-image image.png")
        print("  - With custom parameters: python streamlit_inference.py --mode cli --capacity 20 --steps 30")
        print()
        
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

        # Alzheimer's disease classification (CLI mode)
        if args.alzheimer_image:
            print(f"\n=== ALZHEIMER'S DISEASE CLASSIFICATION ===")
            if not check_alzheimer_model():
                print("❌ Error: Alzheimer model not found or invalid!")
                print("Please ensure 'artifacts/alzheimer_classifier.pkl' exists and contains a valid model.")
            else:
                try:
                    alzheimer_model = load_alzheimer_model()
                    print(f"✅ Loaded Alzheimer model with classes: {alzheimer_model['classes']}")
                    
                    # Check if image file exists
                    if not os.path.exists(args.alzheimer_image):
                        print(f"❌ Error: Image file '{args.alzheimer_image}' not found!")
                    else:
                        # Load and classify image
                        img = load_and_preprocess_image(args.alzheimer_image)
                        if img is None:
                            print(f"❌ Error: Could not load image '{args.alzheimer_image}'")
                        else:
                            # Get target feature count from model
                            target_features = alzheimer_model["scaler"].n_features_in_
                            features = extract_image_features(img, target_features=target_features)
                            if features.size == 0:
                                print(f"❌ Error: Could not extract features from image")
                            else:
                                # Scale features and predict
                                scaler = alzheimer_model["scaler"]
                                features_scaled = scaler.transform(features.reshape(1, -1))
                                
                                classifier = alzheimer_model["classifier"]
                                prediction = classifier.predict(features_scaled)[0]
                                probabilities = classifier.predict_proba(features_scaled)[0]
                                
                                # Get class name and confidence
                                label_encoder = alzheimer_model["label_encoder"]
                                class_name = label_encoder.inverse_transform([prediction])[0]
                                confidence = np.max(probabilities)
                                
                                print(f"📸 Image: {args.alzheimer_image}")
                                print(f"🔍 Prediction: {class_name}")
                                print(f"📊 Confidence: {confidence:.2%}")
                                print(f"📈 All probabilities:")
                                for i, (cls, prob) in enumerate(zip(alzheimer_model["classes"], probabilities)):
                                    marker = "✅" if cls == class_name else "  "
                                    print(f"   {marker} {cls}: {prob:.2%}")
                                
                except Exception as e:
                    print(f"❌ Alzheimer classification failed: {e}")

        # Audio processing (CLI mode)
        if args.audio_files:
            print(f"\n=== AUDIO MFCC ANALYSIS ===")
            if librosa is None:
                print("❌ Error: librosa not available for audio processing!")
                print("Please install librosa: pip install librosa")
            else:
                try:
                    # Create mock file objects for CLI processing
                    class MockAudioFile:
                        def __init__(self, path):
                            self.path = path
                            self.name = os.path.basename(path)
                        
                        def getbuffer(self):
                            with open(self.path, 'rb') as f:
                                return f.read()
                    
                    mock_audio_files = [MockAudioFile(path) for path in args.audio_files]
                    
                    # Extract MFCC features
                    mfcc_features, file_names = extract_mfcc_features(mock_audio_files)
                    
                    if mfcc_features is not None:
                        # Analyze features
                        audio_results = analyze_audio_features(mfcc_features, file_names)
                        if audio_results:
                            print(f"✅ Successfully processed {len(args.audio_files)} audio files!")
                            print(f"📊 MFCC dimensions: {audio_results['mfcc_dimensions']}")
                            print(f"📈 Features per file: {len(audio_results['feature_names'])}")
                            
                            # Display summary
                            print(f"\n📋 Audio Files Summary:")
                            print(audio_results['summary_df'].to_string(index=False))
                            
                            # Save results
                            os.makedirs("artifacts", exist_ok=True)
                            audio_results_path = os.path.join("artifacts", "audio_analysis_results.csv")
                            audio_results['summary_df'].to_csv(audio_results_path, index=False)
                            print(f"💾 Saved audio analysis results to: {audio_results_path}")
                            
                            # Save MFCC features
                            mfcc_path = os.path.join("artifacts", "audio_mfcc_features.npy")
                            np.save(mfcc_path, audio_results['mfcc_features'])
                            print(f"💾 Saved MFCC features to: {mfcc_path}")
                        else:
                            print("❌ Failed to analyze audio features")
                    else:
                        print(f"❌ Audio processing failed: {file_names}")
                        
                except Exception as e:
                    print(f"❌ Audio processing failed: {e}")

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
