"""Text Classification App"""

import streamlit as st
import torch
from typing import Dict
import os

import config
from model_loader import (
    get_device,
    get_model_files,
    load_model,
    predict,
    get_top_prediction,
    get_predictions,
)

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🤖",
    layout="centered"
)

for key in ['model', 'model_loaded', 'device', 'result', 'time']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'model_loaded' else False

if not st.session_state.model_loaded:
    files = get_model_files(config.MODEL_DIR)
    if files:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model(files[0], get_device(), config.NUM_LABELS)
            st.session_state.device = get_device()
            st.session_state.model_loaded = True

st.title(f"{config.APP_ICON} {config.APP_TITLE}")

text = st.text_area("Enter text:", placeholder="Type something...")

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter text")
    elif st.session_state.model:
        with st.spinner("Predicting..."):
            logits, t = predict(st.session_state.model, text, st.session_state.device)
            pred, conf = get_top_prediction(logits, config.LABEL_NAMES)
            probs = get_predictions(logits, config.LABEL_NAMES)
            st.session_state.result = (pred, conf, probs)
            st.session_state.time = t

if st.session_state.result:
    pred, conf, probs = st.session_state.result
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", pred.upper())
    with col2:
        st.metric("Confidence", f"{conf*100:.1f}%")
    st.bar_chart(probs, height=250)
