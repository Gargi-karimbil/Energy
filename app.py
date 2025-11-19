# app.py
# Energy Consumption Forecast (LSTM + local XAI) - updated inputs (NSM raw seconds, Load_Type_num 0/1/2)

import streamlit as st
import os, time, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

st.set_page_config(page_title="Energy Consumption Forecast", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .main { background: linear-gradient(180deg,#f7fbff,#eef6ff); }
      .card { background: white; padding:20px; border-radius:12px;
              box-shadow: 0 8px 24px rgba(9,30,66,0.08); margin-bottom:20px; }
      .title { font-size:28px; color:#0b3d91; font-weight:700; margin-bottom:8px; }
      .subtitle { color:#1453a1; margin-bottom:20px; font-size:16px; }
      .result { background:#f3f8ff; padding:20px; border-radius:10px; margin:15px 0; }
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

# ---------- Auto-find model and data ----------
BASE = "."
model_file = None
pkl_file = None

model_priority = ["bidirectional_w16", "bidirectional_w32", "bidirectional", "lstm", "model"]
for pattern in model_priority:
    for f in os.listdir(BASE):
        if f.lower().startswith(pattern) and f.lower().endswith((".h5", ".keras", ".hdf5")):
            model_file = os.path.join(BASE, f)
            st.sidebar.success(f"✅ Model found: {f}")
            break
    if model_file:
        break

for f in os.listdir(BASE):
    if "all_windows_scaled" in f.lower() and f.lower().endswith(".pkl"):
        pkl_file = os.path.join(BASE, f)
        st.sidebar.success(f"✅ Data found: {f}")
        break

if not model_file:
    st.sidebar.error("❌ No model file found")
    st.error("Model file not found in the app folder. Place the model (.h5/.keras/.hdf5) here.")
    st.stop()

if not pkl_file:
    st.sidebar.error("❌ No preprocessing pickle found")
    st.error("Preprocessing file `all_windows_scaled.pkl` not found in the app folder.")
    st.stop()

st.sidebar.write(f"📁 Model: {os.path.basename(model_file)}")
st.sidebar.write(f"📊 Data: {os.path.basename(pkl_file)}")
st.sidebar.write(f"💾 Model size: {os.path.getsize(model_file) / 1024 / 1024:.1f} MB")

# ---------- Model loading helper ----------
@st.cache_resource
def load_energy_model(model_path):
    model = None
    errors = []
    try:
        model = load_model(model_path, compile=True)
        return model, "Standard load with compile=True"
    except Exception as e:
        errors.append(f"Standard load: {e}")
    try:
        model = load_model(model_path, compile=False)
        return model, "Load with compile=False"
    except Exception as e:
        errors.append(f"Load no compile: {e}")
    try:
        model = load_model(model_path, compile=False, custom_objects={'Bidirectional': Bidirectional, 'LSTM': LSTM})
        return model, "Load with custom objects"
    except Exception as e:
        errors.append(f"Load custom objects: {e}")
    try:
        model = tf.keras.models.load_model(model_path)
        return model, "tf.keras.load_model"
    except Exception as e:
        errors.append(f"tf.keras.load_model: {e}")
    # last resort: try architecture reconstruction (weights loads)
    try:
        architectures = [
            Sequential([Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, 8)), Bidirectional(LSTM(32)), Dense(16, activation='relu'), Dense(1)]),
            Sequential([LSTM(50, return_sequences=True, input_shape=(None, 8)), LSTM(25), Dense(10, activation='relu'), Dense(1)]),
            Sequential([Bidirectional(LSTM(50, input_shape=(None, 8))), Dense(25, activation='relu'), Dense(1)])
        ]
        for i, arch in enumerate(architectures):
            try:
                arch.load_weights(model_path)
                return arch, f"Reconstructed arch {i+1}"
            except Exception:
                continue
    except Exception as e:
        errors.append(f"Architecture reconstruction: {e}")
    return None, errors

with st.spinner("Loading model..."):
    model, load_method = load_energy_model(model_file)

if model is None:
    st.error("All model loading attempts failed.")
    st.code(load_method if isinstance(load_method, list) else str(load_method))
    st.stop()

st.sidebar.success(f"✅ Load method: {load_method}")

# ---------- Load preprocessing data ----------
try:
    with open(pkl_file, "rb") as f:
        all_windows = pickle.load(f)
    # pick W (prefer 16)
    W = 16 if 16 in all_windows else sorted(all_windows.keys())[-1]
    data = all_windows[W]
    X_train = data.get("X_train", None)
    scaler_X = data.get("scaler_X", None)
    scaler_y = data.get("scaler_y", None)
    feature_cols = data.get("feature_cols", None)
    st.sidebar.success(f"✅ Data loaded (W={W})")
except Exception as e:
    st.error(f"Failed to load preprocessing data: {e}")
    st.stop()

# ---------- model input shape inference ----------
try:
    if hasattr(model, 'input_shape'):
        inp_shape = model.input_shape
        if isinstance(inp_shape, list):
            inp_shape = inp_shape[0]
        _, T_model, F_model = inp_shape
    else:
        T_model = X_train.shape[1] if X_train is not None else 16
        F_model = X_train.shape[2] if X_train is not None else 8
    T_model = int(T_model)
    F_model = int(F_model)
except Exception:
    T_model, F_model = 16, 8

st.sidebar.write(f"📐 Input shape: ({T_model}, {F_model})")

# ---------- default feature list ----------
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
    feature_cols = default_features[:F_model]

# Ensure feature_cols length == F_model
if len(feature_cols) > F_model:
    feature_cols = feature_cols[:F_model]
elif len(feature_cols) < F_model:
    feature_cols = feature_cols + [f"feature_{i}" for i in range(len(feature_cols), F_model)]

# ---------- Compute scaled stats and try recover raw mins/maxs ----------
if X_train is not None:
    try:
        X_collapsed = X_train.reshape(-1, X_train.shape[-1])
        feat_means = {feature_cols[i]: float(np.nanmean(X_collapsed[:, i])) for i in range(F_model)}
        feat_stds  = {feature_cols[i]: float(np.nanstd(X_collapsed[:, i]) + 1e-9) for i in range(F_model)}
        feat_mins  = {feature_cols[i]: float(np.nanmin(X_collapsed[:, i])) for i in range(F_model)}
        feat_maxs  = {feature_cols[i]: float(np.nanmax(X_collapsed[:, i])) for i in range(F_model)}
    except Exception:
        feat_means = {col: 0.0 for col in feature_cols}
        feat_stds  = {col: 1.0 for col in feature_cols}
        feat_mins  = {col: 0.0 for col in feature_cols}
        feat_maxs  = {col: 1.0 for col in feature_cols}
else:
    feat_means = {col: 0.0 for col in feature_cols}
    feat_stds  = {col: 1.0 for col in feature_cols}
    feat_mins  = {col: 0.0 for col in feature_cols}
    feat_maxs  = {col: 1.0 for col in feature_cols}

# Try to extract raw min/max from scaler_X (if MinMaxScaler was used and saved)
raw_mins = {col: None for col in feature_cols}
raw_maxs = {col: None for col in feature_cols}
if scaler_X is not None:
    try:
        data_min = getattr(scaler_X, "data_min_", None)
        data_max = getattr(scaler_X, "data_max_", None)
        if data_min is not None and data_max is not None and len(data_min) >= F_model:
            for i, col in enumerate(feature_cols):
                raw_mins[col] = float(data_min[i])
                raw_maxs[col] = float(data_max[i])
    except Exception:
        pass

# ---------- preferred inputs for UI (we present these five in the app) ----------
preferred_features = ["CO2(tCO2)", "NSM", "Leading_Current_Power_Factor", "Load_Type_num", "Lagging_Current_Reactive.Power_kVarh"]
input_features = [f for f in preferred_features if f in feature_cols]

# fill to at most 5 features
for f in feature_cols:
    if f not in input_features and len(input_features) < 5:
        input_features.append(f)

# ---------- Main UI: Input Features ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔧 Input Features")
col1, col2 = st.columns(2)
inputs = {}

for i, fname in enumerate(input_features):
    with (col1 if i % 2 == 0 else col2):
        # Special handling for the five main features
        if fname == "NSM":
            # determine raw bounds from scaler if available; otherwise full day range
            idx = feature_cols.index("NSM")
            if raw_mins.get("NSM") is not None and raw_maxs.get("NSM") is not None:
                raw_min = int(max(0, min(86400, raw_mins["NSM"])))
                raw_max = int(max(0, min(86400, raw_maxs["NSM"])))
                if raw_max <= raw_min:
                    raw_min, raw_max = 0, 86400
            else:
                raw_min, raw_max = 0, 86400

            # default: try invert scaled mean to raw (if scaler exists)
            default_val = 3600
            try:
                if scaler_X is not None:
                    scaled_vec = np.array([feat_means.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
                    inv = scaler_X.inverse_transform(scaled_vec)
                    default_val = int(round(inv[0, idx]))
            except Exception:
                default_val = int(round(feat_means.get("NSM", 3600)))
            default_val = int(min(max(default_val, raw_min), raw_max))

            nsm_val = st.number_input(
                label="**NSM (Seconds from Midnight)**",
                value=default_val,
                min_value=raw_min,
                max_value=raw_max,
                step=1,
                help="Enter seconds from midnight (0 - 86400). Example: 3600 = 1:00 AM, 43200 = 12:00 PM"
            )
            inputs[fname] = int(nsm_val)

        elif fname == "Load_Type_num":
            # enforce categories 0/1/2 (Light, Medium, Heavy)
            # default from raw_mins/raw_maxs or dataset mean
            mean_val = feat_means.get("Load_Type_num", 1.0)
            default_choice = int(round(mean_val)) if mean_val is not None else 1
            default_choice = min(max(default_choice, 0), 2)
            choice = st.selectbox(
                label="**Load_Type (0=Light, 1=Medium, 2=Heavy)**",
                options=[0, 1, 2],
                index=[0,1,2].index(default_choice),
                help="Select encoded load type"
            )
            inputs[fname] = int(choice)

        elif fname == "CO2(tCO2)":
            # continuous, allow 0-10 (safe)
            raw_min = raw_mins.get(fname)
            raw_max = raw_maxs.get(fname)
            if raw_min is None or raw_max is None or raw_min >= raw_max:
                raw_min, raw_max = 0.0, 10.0
            # default try inverse transform
            default_val = feat_means.get(fname, 0.0)
            try:
                if scaler_X is not None:
                    scaled_vec = np.array([feat_means.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
                    inv = scaler_X.inverse_transform(scaled_vec)
                    default_val = float(inv[0, feature_cols.index(fname)])
            except Exception:
                default_val = float(feat_means.get(fname, 0.0))
            val = st.number_input(
                label=f"**{fname}**",
                value=float(round(default_val, 3)),
                min_value=float(raw_min),
                max_value=float(raw_max),
                step=0.01,
                format="%.3f",
                help=f"CO2 (tCO2). Suggested range: {raw_min:.3f} - {raw_max:.3f}"
            )
            inputs[fname] = float(val)

        elif fname == "Leading_Current_Power_Factor":
            # percent 0-100
            raw_min = raw_mins.get(fname)
            raw_max = raw_maxs.get(fname)
            if raw_min is None or raw_max is None or raw_min >= raw_max:
                raw_min, raw_max = 0.0, 100.0
            default_val = feat_means.get(fname, 50.0)
            try:
                if scaler_X is not None:
                    scaled_vec = np.array([feat_means.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
                    inv = scaler_X.inverse_transform(scaled_vec)
                    default_val = float(inv[0, feature_cols.index(fname)])
            except Exception:
                default_val = float(feat_means.get(fname, 50.0))
            val = st.number_input(
                label=f"**{fname} (%)**",
                value=float(round(default_val, 2)),
                min_value=float(raw_min),
                max_value=float(raw_max),
                step=0.1,
                format="%.2f",
                help=f"Power factor (0-100%). Suggested range: {raw_min:.2f} - {raw_max:.2f}"
            )
            inputs[fname] = float(val)

        elif fname == "Lagging_Current_Reactive.Power_kVarh":
            # reactive power: sensible 0-20
            raw_min = raw_mins.get(fname)
            raw_max = raw_maxs.get(fname)
            if raw_min is None or raw_max is None or raw_min >= raw_max:
                raw_min, raw_max = 0.0, 20.0
            default_val = feat_means.get(fname, 0.0)
            try:
                if scaler_X is not None:
                    scaled_vec = np.array([feat_means.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
                    inv = scaler_X.inverse_transform(scaled_vec)
                    default_val = float(inv[0, feature_cols.index(fname)])
            except Exception:
                default_val = float(feat_means.get(fname, 0.0))
            val = st.number_input(
                label=f"**{fname} (kVarh)**",
                value=float(round(default_val, 3)),
                min_value=float(raw_min),
                max_value=float(raw_max),
                step=0.1,
                format="%.3f",
                help=f"Reactive power (kVarh). Suggested range: {raw_min:.3f} - {raw_max:.3f}"
            )
            inputs[fname] = float(val)

        else:
            # generic fallback numeric field
            raw_min = raw_mins.get(fname) if raw_mins.get(fname) is not None else feat_mins.get(fname, 0.0)
            raw_max = raw_maxs.get(fname) if raw_maxs.get(fname) is not None else feat_maxs.get(fname, 1.0)
            default_val = feat_means.get(fname, 0.0)
            val = st.number_input(
                label=f"**{fname}**",
                value=float(round(default_val, 3)),
                min_value=float(raw_min) if raw_min is not None else float(default_val) - 10.0,
                max_value=float(raw_max) if raw_max is not None else float(default_val) + 10.0,
                step=0.1,
                format="%.3f"
            )
            inputs[fname] = float(val)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prediction function ----------
def make_prediction(feature_inputs):
    # construct full vector
    full_vector = np.zeros(F_model, dtype=float)
    for i, col in enumerate(feature_cols):
        if col in feature_inputs:
            full_vector[i] = float(feature_inputs[col])
        else:
            full_vector[i] = float(feat_means.get(col, 0.0))

    # scale features using scaler_X if available
    input_2d = full_vector.reshape(1, -1)
    if scaler_X is not None:
        try:
            scaled_input = scaler_X.transform(input_2d)
        except Exception:
            scaled_input = np.array([(input_2d[0, j] - feat_means[feature_cols[j]]) / (feat_stds[feature_cols[j]] + 1e-9) for j in range(F_model)]).reshape(1, -1)
    else:
        scaled_input = np.array([(input_2d[0, j] - feat_means[feature_cols[j]]) / (feat_stds[feature_cols[j]] + 1e-9) for j in range(F_model)]).reshape(1, -1)

    # repeat to timesteps
    input_3d = np.repeat(scaled_input[:, np.newaxis, :], T_model, axis=1)

    # predict
    pred_scaled = model.predict(input_3d, verbose=0)
    if scaler_y is not None:
        try:
            pred = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])
        except Exception:
            pred = float(pred_scaled[0, 0])
    else:
        pred = float(pred_scaled[0, 0])

    return pred, input_3d, full_vector

# ---------- simple XAI: leave-one-mean-out impacts ----------
def compute_feature_impacts(base_prediction, input_3d, full_vector):
    impacts = []
    mean_vector = np.array([feat_means[col] for col in feature_cols])
    if scaler_X is not None:
        try:
            mean_scaled = scaler_X.transform(mean_vector.reshape(1, -1)).flatten()
        except Exception:
            mean_scaled = np.zeros(F_model)
    else:
        mean_scaled = np.zeros(F_model)

    for i, col in enumerate(feature_cols):
        modified = input_3d.copy()
        modified[0, :, i] = mean_scaled[i]
        try:
            pmod = model.predict(modified, verbose=0)
            if scaler_y is not None:
                try:
                    pmod_raw = float(scaler_y.inverse_transform(pmod.reshape(-1, 1))[0, 0])
                except Exception:
                    pmod_raw = float(pmod[0, 0])
            else:
                pmod_raw = float(pmod[0, 0])
            impact = base_prediction - pmod_raw
            impacts.append((col, impact, full_vector[i], feat_means[col]))
        except Exception:
            impacts.append((col, 0.0, full_vector[i], feat_means[col]))

    df_impacts = pd.DataFrame(impacts, columns=["feature", "effect_kWh", "input_val", "mean_val"])
    df_impacts["abs_effect"] = df_impacts["effect_kWh"].abs()
    df_impacts = df_impacts.sort_values("abs_effect", ascending=False)
    return df_impacts

# ---------- Predict button ----------
if st.button("🚀 Predict Energy Consumption", type="primary", use_container_width=True):
    with st.spinner("Predicting and computing feature impacts..."):
        try:
            start = time.time()
            prediction, input_3d, full_vector = make_prediction(inputs)
            pred_time = time.time() - start

            start_xai = time.time()
            df_impacts = compute_feature_impacts(prediction, input_3d, full_vector)
            xai_time = time.time() - start_xai

            # display result
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="result">', unsafe_allow_html=True)
            st.markdown("## 📊 Prediction Result")
            st.markdown(f"# **{prediction:.2f} kWh**")
            st.caption(f"Prediction time: {pred_time:.2f}s | XAI analysis: {xai_time:.2f}s")
            st.markdown('</div>', unsafe_allow_html=True)

            # XAI explanation
            st.markdown("## 🔍 Explanation")
            if not df_impacts.empty:
                top = df_impacts.iloc[0]
                feat_name = top["feature"]
                impact_val = top["effect_kWh"]
                input_val = top["input_val"]
                mean_val = top["mean_val"]
                std_val = feat_stds.get(feat_name, 1.0)
                if input_val > mean_val + 0.5 * std_val:
                    rel = "higher than average"
                elif input_val < mean_val - 0.5 * std_val:
                    rel = "lower than average"
                else:
                    rel = "near average"
                dir_text = "increases" if impact_val > 0 else "decreases"
                st.markdown(f"The prediction is most influenced by **{feat_name}**. Your input value (**{input_val:.2f}**) is **{rel}** (dataset mean: {mean_val:.2f}), which **{dir_text}** energy consumption by approximately **{abs(impact_val):.2f} kWh** compared to the mean value.")

            # show selected input features impacts in table
            st.markdown("### 📈 Feature Impacts (Selected Inputs)")
            filtered_rows = []
            for f in input_features:
                row = df_impacts[df_impacts['feature'] == f]
                if not row.empty:
                    filtered_rows.append(row.iloc[0].to_dict())
                else:
                    idx = feature_cols.index(f) if f in feature_cols else None
                    input_val = float(full_vector[idx]) if idx is not None else 0.0
                    mean_val = float(feat_means.get(f, 0.0))
                    filtered_rows.append({'feature': f, 'effect_kWh': 0.0, 'input_val': input_val, 'mean_val': mean_val, 'abs_effect': 0.0})
            display_impacts = pd.DataFrame(filtered_rows)
            display_impacts["Impact (kWh)"] = display_impacts["effect_kWh"].apply(lambda x: f"{x:+.3f}")
            display_impacts["Your Value"] = display_impacts["input_val"].apply(lambda x: f"{x:.3f}")
            display_impacts["Dataset Mean"] = display_impacts["mean_val"].apply(lambda x: f"{x:.3f}")
            st.dataframe(display_impacts[["feature", "Impact (kWh)", "Your Value", "Dataset Mean"]], use_container_width=True, hide_index=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.markdown("Check input ranges and ensure model + preprocessing pickle are compatible.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;font-size:14px;'>Energy Consumption Forecast • LSTM • Local XAI • Streamlit</div>", unsafe_allow_html=True)
