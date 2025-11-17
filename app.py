# Fully updated working version - Energy Consumption Forecast (LSTM + fast local XAI)

import streamlit as st
import os, time, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

st.set_page_config(page_title="Energy Consumption Forecast", layout="centered")

# ---------- Professional Styling ----------
st.markdown(
    """
    <style>
      .main { background: linear-gradient(180deg,#f7fbff,#eef6ff); }
      .card { background: white; padding:20px; border-radius:12px;
              box-shadow: 0 8px 24px rgba(9,30,66,0.08); margin-bottom:20px; }
      .title { font-size:28px; color:#0b3d91; font-weight:700; margin-bottom:8px; }
      .subtitle { color:#1453a1; margin-bottom:20px; font-size:16px; }
      .result { background:#f3f8ff; padding:20px; border-radius:10px; margin:15px 0; }
      .diagnostic { background:#fff3cd; padding:12px; border-radius:8px; margin:10px 0; 
                    border-left:4px solid #ffc107; }
      .success { background:#d1edff; padding:12px; border-radius:8px; margin:10px 0;
                 border-left:4px solid #0d6efd; }
      .feature-table { font-size:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">⚡ Energy Consumption Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">LSTM Model with Local Explainable AI (XAI)</div>', unsafe_allow_html=True)

# ---------- Diagnostics Sidebar ----------
st.sidebar.markdown("### 🔍 System Diagnostics")
st.sidebar.write(f"TensorFlow: {tf.__version__}")
st.sidebar.write(f"Keras: {tf.keras.__version__}")

# ---------- Auto-find Resources ----------
BASE = "."
model_file = None
pkl_file = None

# Find model file with priority
model_priority = [
    "bidirectional_w16", "bidirectional_w32", "bidirectional",
    "lstm", "model"
]

for pattern in model_priority:
    for f in os.listdir(BASE):
        if f.lower().startswith(pattern) and f.lower().endswith((".h5", ".keras", ".hdf5")):
            model_file = os.path.join(BASE, f)
            st.sidebar.success(f"✅ Model found: {f}")
            break
    if model_file:
        break

# Find pickle file
for f in os.listdir(BASE):
    if "all_windows_scaled" in f.lower() and f.lower().endswith(".pkl"):
        pkl_file = os.path.join(BASE, f)
        st.sidebar.success(f"✅ Data found: {f}")
        break

if not model_file:
    st.sidebar.error("❌ No model file found")
    st.error("""
    **Model file not found!** 
    
    Please ensure you have one of these files in the same folder:
    - `bidirectional_w16.h5` (preferred)
    - `bidirectional.h5` 
    - `lstm_model.h5`
    - Any `.h5`, `.keras`, or `.hdf5` file
    """)
    st.stop()

if not pkl_file:
    st.sidebar.error("❌ No data file found")
    st.error("""
    **Preprocessing data not found!**
    
    Please ensure `all_windows_scaled.pkl` is in the same folder.
    """)
    st.stop()

st.sidebar.write(f"📁 Model: {os.path.basename(model_file)}")
st.sidebar.write(f"📊 Data: {os.path.basename(pkl_file)}")
st.sidebar.write(f"💾 Model size: {os.path.getsize(model_file) / 1024 / 1024:.1f} MB")

# ---------- Advanced Model Loading with Multiple Fallbacks ----------
@st.cache_resource
def load_energy_model(model_path):
    """Load model with multiple fallback strategies"""
    model = None
    errors = []
    
    # Strategy 1: Standard load with compile=True
    try:
        st.sidebar.info("🔄 Attempting standard model load...")
        model = load_model(model_path, compile=True)
        st.sidebar.success("✅ Model loaded with compile=True")
        return model, "Standard load with compilation"
    except Exception as e:
        errors.append(f"Standard load: {str(e)}")
    
    # Strategy 2: Load without compilation
    try:
        st.sidebar.info("🔄 Attempting load without compilation...")
        model = load_model(model_path, compile=False)
        st.sidebar.success("✅ Model loaded with compile=False")
        return model, "Load without compilation"
    except Exception as e:
        errors.append(f"Load without compile: {str(e)}")
    
    # Strategy 3: Load with custom objects
    try:
        st.sidebar.info("🔄 Attempting load with custom objects...")
        model = load_model(model_path, compile=False,
                          custom_objects={
                              'Bidirectional': Bidirectional,
                              'LSTM': LSTM
                          })
        st.sidebar.success("✅ Model loaded with custom objects")
        return model, "Load with custom objects"
    except Exception as e:
        errors.append(f"Load with custom objects: {str(e)}")
    
    # Strategy 4: Try different file formats
    try:
        st.sidebar.info("🔄 Attempting Keras format load...")
        # If it's actually a .keras format but named .h5
        model = tf.keras.models.load_model(model_path)
        st.sidebar.success("✅ Model loaded as Keras format")
        return model, "Keras format load"
    except Exception as e:
        errors.append(f"Keras format: {str(e)}")
    
    # Strategy 5: Build architecture and load weights (last resort)
    try:
        st.sidebar.info("🔄 Attempting architecture reconstruction...")
        # Common LSTM architectures - try different configurations
        architectures = [
            # Architecture 1: Common bidirectional LSTM
            Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, 8)),
                Bidirectional(LSTM(32)),
                Dense(16, activation='relu'),
                Dense(1)
            ]),
            # Architecture 2: Simpler LSTM
            Sequential([
                LSTM(50, return_sequences=True, input_shape=(None, 8)),
                LSTM(25),
                Dense(10, activation='relu'),
                Dense(1)
            ]),
            # Architecture 3: Single bidirectional
            Sequential([
                Bidirectional(LSTM(50, input_shape=(None, 8))),
                Dense(25, activation='relu'),
                Dense(1)
            ])
        ]
        
        for i, arch in enumerate(architectures):
            try:
                arch.load_weights(model_path)
                model = arch
                st.sidebar.success(f"✅ Model reconstructed (arch {i+1})")
                return model, f"Architecture reconstruction {i+1}"
            except:
                continue
                
        errors.append("All architecture reconstructions failed")
    except Exception as e:
        errors.append(f"Architecture reconstruction: {str(e)}")
    
    return None, errors

# Load the model
with st.spinner("Loading model... This may take a moment"):
    model, load_method = load_energy_model(model_file)

if model is None:
    st.error("""
    **❌ All model loading attempts failed!**
    
    **Possible solutions:**
    1. **Retrain the model** with your current TensorFlow version
    2. **Check model compatibility** - ensure it's a valid Keras model
    3. **Try a different model format** (.keras instead of .h5)
    4. **Verify the model file** isn't corrupted
    
    **Technical details:**
    """)
    st.code(load_method if isinstance(load_method, list) else str(load_method))
    st.stop()

st.sidebar.success(f"✅ Load method: {load_method}")

# ---------- Load Preprocessing Data ----------
try:
    with open(pkl_file, "rb") as f:
        all_windows = pickle.load(f)
    
    # Choose appropriate window size
    W = 16 if 16 in all_windows else sorted(all_windows.keys())[-1]
    data = all_windows[W]
    
    X_train = data.get("X_train", None)
    scaler_X = data.get("scaler_X", None)
    scaler_y = data.get("scaler_y", None)
    feature_cols = data.get("feature_cols", None)
    
    st.sidebar.success(f"✅ Data loaded (W={W})")
    
except Exception as e:
    st.error(f"Failed to load preprocessing data: {str(e)}")
    st.stop()

# ---------- Model Configuration ----------
try:
    # Get model input shape
    if hasattr(model, 'input_shape'):
        inp_shape = model.input_shape
        if isinstance(inp_shape, list):
            inp_shape = inp_shape[0]
        _, T_model, F_model = inp_shape
    else:
        # Fallback to data-based inference
        T_model = X_train.shape[1] if X_train is not None else 16
        F_model = X_train.shape[2] if X_train is not None else 8
    
    T_model = int(T_model) if T_model is not None else 16
    F_model = int(F_model) if F_model is not None else 8
    
except Exception as e:
    st.warning(f"Could not determine model shape: {e}")
    T_model, F_model = 16, 8

st.sidebar.write(f"📐 Input shape: ({T_model}, {F_model})")

# ---------- Feature Configuration ----------
default_features = [
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh', 
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM',
    'WeekStatus_num',
    'Load_Type_num'
]

if feature_cols is None:
    feature_cols = default_features

# Ensure feature columns match model dimensions
if len(feature_cols) > F_model:
    feature_cols = feature_cols[:F_model]
elif len(feature_cols) < F_model:
    feature_cols = feature_cols + [f"feature_{i}" for i in range(len(feature_cols), F_model)]

# ---------- Compute Feature Statistics ----------
if X_train is not None:
    try:
        X_collapsed = X_train.reshape(-1, X_train.shape[-1])
        feat_means = {feature_cols[i]: float(np.nanmean(X_collapsed[:, i])) for i in range(F_model)}
        feat_stds = {feature_cols[i]: float(np.nanstd(X_collapsed[:, i]) + 1e-9) for i in range(F_model)}
        feat_mins = {feature_cols[i]: float(np.nanmin(X_collapsed[:, i])) for i in range(F_model)}
        feat_maxs = {feature_cols[i]: float(np.nanmax(X_collapsed[:, i])) for i in range(F_model)}
    except:
        feat_means = {col: 0.0 for col in feature_cols}
        feat_stds = {col: 1.0 for col in feature_cols}
        feat_mins = {col: 0.0 for col in feature_cols}
        feat_maxs = {col: 1.0 for col in feature_cols}
else:
    feat_means = {col: 0.0 for col in feature_cols}
    feat_stds = {col: 1.0 for col in feature_cols}
    feat_mins = {col: 0.0 for col in feature_cols}
    feat_maxs = {col: 1.0 for col in feature_cols}

# ---------- Top Features Selection ----------
preferred_features = ["CO2(tCO2)", "NSM", "Leading_Current_Power_Factor", "Load_Type_num", "Lagging_Current_Reactive.Power_kVarh"]
input_features = [f for f in preferred_features if f in feature_cols]

# Fill remaining slots
for f in feature_cols:
    if f not in input_features and len(input_features) < 5:
        input_features.append(f)

# ---------- Main Input Interface ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔧 Input Features")

# Create two columns for better layout
col1, col2 = st.columns(2)
inputs = {}

for i, fname in enumerate(input_features):
    with col1 if i % 2 == 0 else col2:
        # Set reasonable bounds and step based on feature type
        if "num" in fname.lower() or fname == "NSM":
            # Integer features
            min_val = int(feat_mins.get(fname, 0))
            max_val = int(feat_maxs.get(fname, 100))
            default_val = int(round(feat_means.get(fname, 0)))
            step = 1
            value = st.number_input(
                label=f"**{fname}**",
                value=default_val,
                min_value=min_val,
                max_value=max_val,
                step=step,
                help=f"Range: {min_val} - {max_val}"
            )
        else:
            # Continuous features
            min_val = float(feat_mins.get(fname, 0))
            max_val = float(feat_maxs.get(fname, 10))
            default_val = float(round(feat_means.get(fname, 0), 2))
            step = 0.1
            value = st.number_input(
                label=f"**{fname}**", 
                value=default_val,
                min_value=min_val,
                max_value=max_val,
                step=step,
                format="%.2f",
                help=f"Range: {min_val:.2f} - {max_val:.2f}"
            )
        inputs[fname] = float(value)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prediction Function ----------
def make_prediction(feature_inputs):
    """Make prediction with proper feature scaling"""
    # Build full feature vector
    full_vector = np.zeros(F_model, dtype=float)
    for i, col in enumerate(feature_cols):
        if col in feature_inputs:
            full_vector[i] = feature_inputs[col]
        else:
            full_vector[i] = feat_means.get(col, 0.0)
    
    # Scale features
    input_2d = full_vector.reshape(1, -1)
    
    if scaler_X is not None:
        try:
            scaled_input = scaler_X.transform(input_2d)
        except Exception:
            # Fallback to manual scaling
            scaled_input = np.array([
                (input_2d[0, j] - feat_means[feature_cols[j]]) / (feat_stds[feature_cols[j]] + 1e-9) 
                for j in range(F_model)
            ]).reshape(1, -1)
    else:
        # Manual z-score scaling
        scaled_input = np.array([
            (input_2d[0, j] - feat_means[feature_cols[j]]) / (feat_stds[feature_cols[j]] + 1e-9) 
            for j in range(F_model)
        ]).reshape(1, -1)
    
    # Expand to 3D for LSTM (batch_size, timesteps, features)
    input_3d = np.repeat(scaled_input[:, np.newaxis, :], T_model, axis=1)
    
    # Make prediction
    try:
        prediction_scaled = model.predict(input_3d, verbose=0)
        
        # Inverse transform if scaler available
        if scaler_y is not None:
            try:
                prediction = float(scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0])
            except:
                prediction = float(prediction_scaled[0, 0])
        else:
            prediction = float(prediction_scaled[0, 0])
            
        return prediction, input_3d, full_vector
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

# ---------- XAI Analysis ----------
def compute_feature_impacts(base_prediction, input_3d, full_vector):
    """Compute feature importance using leave-one-mean-out"""
    impacts = []
    
    # Get mean vector for comparison
    mean_vector = np.array([feat_means[col] for col in feature_cols])
    
    if scaler_X is not None:
        try:
            mean_scaled = scaler_X.transform(mean_vector.reshape(1, -1)).flatten()
        except:
            mean_scaled = np.zeros(F_model)
    else:
        mean_scaled = np.zeros(F_model)
    
    for i, col in enumerate(feature_cols):
        # Replace feature with mean and predict
        modified_input = input_3d.copy()
        modified_input[0, :, i] = mean_scaled[i]
        
        try:
            pred_modified = model.predict(modified_input, verbose=0)
            
            if scaler_y is not None:
                try:
                    pred_modified_raw = float(scaler_y.inverse_transform(pred_modified.reshape(-1, 1))[0, 0])
                except:
                    pred_modified_raw = float(pred_modified[0, 0])
            else:
                pred_modified_raw = float(pred_modified[0, 0])
            
            # Impact = change from base prediction
            impact = base_prediction - pred_modified_raw
            impacts.append((col, impact, full_vector[i], feat_means[col]))
            
        except Exception as e:
            st.warning(f"Could not compute impact for {col}: {e}")
            impacts.append((col, 0.0, full_vector[i], feat_means[col]))
    
    # Create results dataframe
    df_impacts = pd.DataFrame(impacts, columns=["feature", "effect_kWh", "input_val", "mean_val"])
    df_impacts["abs_effect"] = df_impacts["effect_kWh"].abs()
    df_impacts = df_impacts.sort_values("abs_effect", ascending=False)
    
    return df_impacts

# ---------- Prediction Trigger ----------
if st.button("🚀 Predict Energy Consumption", type="primary", use_container_width=True):
    with st.spinner("Making prediction and analyzing feature impacts..."):
        try:
            start_time = time.time()
            
            # Make prediction
            prediction, input_3d, full_vector = make_prediction(inputs)
            prediction_time = time.time() - start_time
            
            # Compute feature impacts
            impact_start = time.time()
            df_impacts = compute_feature_impacts(prediction, input_3d, full_vector)
            impact_time = time.time() - impact_start
            
            # ---------- Display Results ----------
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Prediction Result
            st.markdown('<div class="result">', unsafe_allow_html=True)
            st.markdown(f"## 📊 Prediction Result")
            st.markdown(f"# **{prediction:.2f} kWh**")
            st.caption(f"Prediction time: {prediction_time:.2f}s | XAI analysis: {impact_time:.2f}s")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # XAI Explanation
            st.markdown("## 🔍 Explanation")
            
            if not df_impacts.empty:
                top_feature = df_impacts.iloc[0]
                feat_name = top_feature["feature"]
                impact_val = top_feature["effect_kWh"]
                input_val = top_feature["input_val"]
                mean_val = top_feature["mean_val"]
                
                # Determine relationship to mean
                std_val = feat_stds.get(feat_name, 1.0)
                if input_val > mean_val + 0.5 * std_val:
                    relationship = "higher than average"
                elif input_val < mean_val - 0.5 * std_val:
                    relationship = "lower than average" 
                else:
                    relationship = "near average"
                
                direction = "increases" if impact_val > 0 else "decreases"
                
                st.markdown(f"""
                The prediction is most influenced by **{feat_name}**. 
                
                Your input value (**{input_val:.2f}**) is **{relationship}** (dataset mean: {mean_val:.2f}), 
                which **{direction}** energy consumption by approximately **{abs(impact_val):.2f} kWh** 
                compared to using the average value.
                """)
            
            # ----------------- UPDATED: Show ONLY the 5 input features in the table -----------------
            st.markdown("### 📈 Feature Impacts (Selected Inputs)")

            # Build a filtered dataframe that preserves the order of input_features
            filtered_rows = []
            for f in input_features:
                row = df_impacts[df_impacts['feature'] == f]
                if not row.empty:
                    filtered_rows.append(row.iloc[0].to_dict())
                else:
                    # Feature not present in df_impacts (unlikely) -> add placeholder
                    idx = feature_cols.index(f) if f in feature_cols else None
                    input_val = float(full_vector[idx]) if idx is not None else 0.0
                    mean_val = float(feat_means.get(f, 0.0))
                    filtered_rows.append({
                        'feature': f,
                        'effect_kWh': 0.0,
                        'input_val': input_val,
                        'mean_val': mean_val,
                        'abs_effect': 0.0
                    })

            display_impacts = pd.DataFrame(filtered_rows)
            display_impacts["Impact (kWh)"] = display_impacts["effect_kWh"].apply(lambda x: f"{x:+.3f}")
            display_impacts["Your Value"] = display_impacts["input_val"].apply(lambda x: f"{x:.3f}")
            display_impacts["Dataset Mean"] = display_impacts["mean_val"].apply(lambda x: f"{x:.3f}")

            st.dataframe(
                display_impacts[["feature", "Impact (kWh)", "Your Value", "Dataset Mean"]],
                use_container_width=True,
                hide_index=True
            )
            # -----------------------------------------------------------------------------------------
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.markdown("""
            **Troubleshooting tips:**
            - Check that all input values are within reasonable ranges
            - Try adjusting input values closer to the dataset means
            - Ensure the model file is compatible with your TensorFlow version
            """)

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "Energy Consumption Forecast • LSTM Model • Local XAI • Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
