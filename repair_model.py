# app.py
# Simple professional Streamlit app for: Energy Consumption Forecast + fast local XAI
# Put this file in the same folder as your model (.keras or .h5) and all_windows_scaled.pkl
# Run: streamlit run app.py

import os
import time
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Energy Consumption Forecast", layout="centered")

# ---- small CSS to make it look professional ----
st.markdown(
    """
    <style>
    .app-title { font-size:28px; color:#0b3d91; font-weight:700; margin-bottom:6px; }
    .app-sub { color:#1453a1; margin-bottom:18px; }
    .card { background:white; padding:14px; border-radius:12px; box-shadow:0 6px 22px rgba(11,61,145,0.06); }
    .predict-btn { background-color:#0b61ff; color:white; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Energy Consumption Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Enter values for the most important features (as in the paper) and click <b>Predict</b>.</div>', unsafe_allow_html=True)

BASE = "."

# ----------------- helpers & caching -----------------
@st.cache_resource
def cached_load_model(model_path: str):
    """Load Keras model (compile=False). Cached so it isn't reloaded every interaction."""
    try:
        m = load_model(model_path, compile=False)
        return m, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def cached_load_pickle(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def auto_find_files(base: str = "."):
    model_file = None
    keras_file = None
    for f in os.listdir(base):
        if f.lower().endswith(".keras"):
            keras_file = os.path.join(base, f)
            break
    if keras_file:
        model_file = keras_file
    else:
        # prefer bidirectional W16 .h5 then any bidirectional .h5
        for f in os.listdir(base):
            if f.lower().endswith(".h5") and "bidirectional" in f.lower() and "w16" in f.lower():
                model_file = os.path.join(base, f)
                break
        if model_file is None:
            for f in os.listdir(base):
                if f.lower().endswith(".h5") and "bidirectional" in f.lower():
                    model_file = os.path.join(base, f)
                    break
    pkl_file = None
    for f in os.listdir(base):
        if f.lower().startswith("all_windows_scaled") and f.lower().endswith(".pkl"):
            pkl_file = os.path.join(base, f)
            break
    return model_file, pkl_file

def ensure_feature_order(feature_cols: List[str], model_F: int) -> List[str]:
    cols = feature_cols.copy()
    if len(cols) < model_F:
        # pad generic names
        cols += [f"feat_{i}" for i in range(len(cols), model_F)]
    elif len(cols) > model_F:
        cols = cols[:model_F]
    return cols

def build_input_vector_from_top_inputs(feature_cols: List[str], input_top: Dict[str, float], feat_means: Dict[str,float], F_model: int):
    vec = np.zeros(F_model, dtype=float)
    for i, fc in enumerate(feature_cols):
        if fc in input_top:
            vec[i] = input_top[fc]
        else:
            vec[i] = feat_means.get(fc, 0.0)
    return vec.reshape(1, -1)

# ----------------- auto-find model & pickle -----------------
MODEL_FILE, PKL_FILE = auto_find_files(BASE)
if not MODEL_FILE or not PKL_FILE:
    st.error("Could not auto-find the model (.keras or .h5) and/or preprocessing pickle (all_windows_scaled.pkl) in this folder.")
    if MODEL_FILE:
        st.info(f"Found model: {os.path.basename(MODEL_FILE)}")
    else:
        st.info("No .keras/.h5 model found. Put model.keras or .h5 in this folder.")
    if PKL_FILE:
        st.info(f"Found pickle: {os.path.basename(PKL_FILE)}")
    else:
        st.info("No all_windows_scaled.pkl found. Put that pickle in this folder.")
    st.stop()

# try to load model
model, model_err = cached_load_model(MODEL_FILE)
if model is None:
    st.error(f"Failed to load model ({os.path.basename(MODEL_FILE)}).")
    st.write("Error:", model_err)
    st.write("If your model was repaired to .keras, name it <model>.keras or try to re-save from training Colab as modern format.")
    st.stop()

# try to load pickle
try:
    all_windows = cached_load_pickle(PKL_FILE)
except Exception as e:
    st.error("Failed to load preprocessing pickle: " + str(e))
    st.stop()

# choose window (paper used W=16)
W = 16 if 16 in all_windows else sorted(all_windows.keys())[-1]
data = all_windows[W]

X_train = data.get("X_train", None)    # (ns, T, F)
scaler_X = data.get("scaler_X", None)
scaler_y = data.get("scaler_y", None)
feature_cols = data.get("feature_cols", None)

default_feature_cols = [
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
    feature_cols = default_feature_cols.copy()

# infer model input shape
try:
    inp = model.input_shape
    if isinstance(inp, list):
        inp = inp[0]
    _, T_model, F_model = inp
    T_model = int(T_model or (X_train.shape[1] if X_train is not None else 16))
    F_model = int(F_model or (X_train.shape[2] if X_train is not None else len(feature_cols)))
except Exception:
    T_model = X_train.shape[1] if X_train is not None else 16
    F_model = X_train.shape[2] if X_train is not None else len(feature_cols)

feature_cols = ensure_feature_order(feature_cols, F_model)

# compute dataset collapsed means for quick filling and local XAI baseline
if X_train is not None:
    X_collapsed = np.mean(X_train, axis=1)  # shape (ns, F)
    feat_means = {feature_cols[i]: float(np.mean(X_collapsed[:, i])) for i in range(F_model)}
    feat_stds = {feature_cols[i]: float(np.std(X_collapsed[:, i]) + 1e-9) for i in range(F_model)}
else:
    feat_means = {c: 0.0 for c in feature_cols}
    feat_stds = {c: 1.0 for c in feature_cols}

# pick top-5 features to show (paper-specified important)
preferred_top5 = ["CO2(tCO2)", "NSM", "Leading_Current_Power_Factor", "Load_Type_num", "Lagging_Current_Reactive.Power_kVarh"]
top5 = [c for c in preferred_top5 if c in feature_cols]
# fill to 5 if needed
for c in feature_cols:
    if c not in top5:
        top5.append(c)
    if len(top5) == 5:
        break

# --------- UI: show input fields (center card) ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Input (top features)")
input_vals = {}
cols = st.columns(2)
for i, fname in enumerate(top5):
    with cols[i % 2]:
        default_val = feat_means.get(fname, 0.0)
        if fname in ("NSM", "WeekStatus_num", "Load_Type_num"):
            v = st.number_input(fname, value=int(round(default_val)), step=1, format="%d", key=f"inp_{fname}")
        else:
            v = st.number_input(fname, value=float(round(default_val, 4)), key=f"inp_{fname}")
        input_vals[fname] = float(v)
st.markdown("</div>", unsafe_allow_html=True)

# Predict button inside a small form to prevent auto-run
predict_clicked = st.button("Predict", key="predict_btn")

# Run prediction when clicked
if predict_clicked:
    # build full vector in feature_cols order
    full_vec_2d = build_input_vector_from_top_inputs(feature_cols, input_vals, feat_means, F_model)  # shape (1, F_model)

    # scale features to training scaling
    if scaler_X is not None:
        try:
            arr_scaled = scaler_X.transform(full_vec_2d)
        except Exception as e:
            # fallback: z-score by dataset mean/std
            arr_scaled = (full_vec_2d - np.array([feat_means[c] for c in feature_cols]).reshape(1, -1)) / \
                         np.array([feat_stds[c] for c in feature_cols]).reshape(1, -1)
    else:
        arr_scaled = (full_vec_2d - np.array([feat_means[c] for c in feature_cols]).reshape(1, -1)) / \
                     np.array([feat_stds[c] for c in feature_cols]).reshape(1, -1)

    # expand to timesteps expected by the model
    x3d = np.repeat(arr_scaled[:, np.newaxis, :], T_model, axis=1)   # shape (1, T_model, F_model)

    # model predict
    try:
        t0 = time.time()
        y_s = model.predict(x3d, verbose=0).reshape(-1, 1)
        elapsed = time.time() - t0
    except Exception as e:
        st.error("Prediction failed: " + str(e))
        raise

    # inverse scale for y
    if scaler_y is not None:
        try:
            y_raw = float(scaler_y.inverse_transform(y_s).reshape(-1)[0])
        except Exception:
            y_raw = float(y_s[0, 0])
    else:
        y_raw = float(y_s[0, 0])

    # FAST local XAI: set each feature to mean (in scaled space) and measure change
    # Build baseline scaled mean vector
    if scaler_X is not None:
        try:
            mean_vec_2d = np.array([feat_means[c] for c in feature_cols]).reshape(1, -1)
            mean_scaled = scaler_X.transform(mean_vec_2d).reshape(-1)  # scaled baseline
        except Exception:
            mean_scaled = np.zeros(F_model)
    else:
        mean_scaled = np.zeros(F_model)

    impacts = []
    for i, fname in enumerate(feature_cols):
        x_test = x3d.copy()
        # set column i to baseline (scaled)
        x_test[0, :, i] = mean_scaled[i]
        try:
            p = model.predict(x_test, verbose=0).reshape(-1, 1)
            p_raw = float(scaler_y.inverse_transform(p).reshape(-1)[0]) if scaler_y is not None else float(p[0, 0])
        except Exception:
            p_raw = float(model.predict(x_test, verbose=0).reshape(-1)[0])
        impact = y_raw - p_raw  # positive => feature increases prediction
        impacts.append((fname, impact))

    df_imp = pd.DataFrame(impacts, columns=["feature", "effect_kWh"])
    # sort by absolute effect descending
    df_imp["abs_effect"] = df_imp["effect_kWh"].abs()
    df_imp = df_imp.sort_values("abs_effect", ascending=False).reset_index(drop=True)

    # ------- Present result -------
    st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("### Prediction")
    st.metric(label="Predicted energy (next timestep)", value=f"{y_raw:.3f} kWh")
    st.markdown(f"<div style='color: #666; font-size:12px;'>Computation time: {elapsed:.2f}s</div>", unsafe_allow_html=True)

    # Friendly natural-language why
    # top contributor:
    top1 = df_imp.iloc[0]
    direction = "increases" if top1["effect_kWh"] > 0 else "decreases"
    # comment whether input value is higher/lower than mean by z-score:
    top_feat = top1["feature"]
    top_val = float(full_vec_2d[0, feature_cols.index(top_feat)])
    meanv = feat_means.get(top_feat, 0.0)
    stdv = feat_stds.get(top_feat, 1.0)
    z = (top_val - meanv) / (stdv + 1e-9)
    if abs(z) < 1:
        magnitude_text = "close to the dataset average"
    elif abs(z) < 2:
        magnitude_text = "moderately different from the dataset average"
    else:
        magnitude_text = "substantially different from the dataset average"

    explanation = (
        f"**Why (short):** The prediction is primarily influenced by **{top_feat}**. "
        f"Your input for {top_feat} is {magnitude_text} (z={z:.2f}), so it {direction} the forecast by about {abs(top1['effect_kWh']):.2f} kWh."
    )
    st.markdown(explanation, unsafe_allow_html=True)

    # show table of top contributors (top 5)
    show = df_imp[["feature", "effect_kWh"]].head(5).copy()
    total = show["effect_kWh"].abs().sum() + 1e-9
    show["% contribution (local)"] = (show["effect_kWh"].abs() / total * 100).round(1)
    show = show.rename(columns={"effect_kWh": "Effect (kWh)"})
    # present the table (avoid pandas .style which needs jinja2)
    st.markdown("#### Top feature effects (local)")
    st.table(show)

    st.markdown("</div>", unsafe_allow_html=True)
