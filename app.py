"""
Streamlit Text Classification Web Application
Ứng dụng web dùng Streamlit để load và chạy inference với model PyTorch
"""

import streamlit as st
import torch
import time
import traceback
from typing import Optional, Tuple, List, Dict
import os
import sys

# Import các module từ project
try:
    import config
    from model_loader import (
        get_device,
        get_model_files,
        get_model_file_size,
        load_model,
        predict,
        get_predictions,
        get_top_prediction,
        preprocess_text,
        DummyTextClassifier
    )
except ImportError as e:
    st.error(f"Lỗi import module: {str(e)}")
    st.stop()


# =============================================================================
# CẤU HÌNH TRANG
# =============================================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS
# =============================================================================
def load_custom_css():
    """Load custom CSS styling"""
    st.markdown("""
    <style>
    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .header-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.5em;
    }
    .header-banner p {
        color: #f0f0f0;
        margin: 10px 0 0 0;
        font-size: 1.1em;
    }
    
    /* Rounded buttons */
    .stButton > button {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    /* Colored headers */
    .section-header {
        color: #667eea;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 2px solid #667eea;
        padding-bottom: 5px;
    }
    
    /* Success/Error boxes */
    .success-box {
        padding: 15px;
        background-color: #d4edda;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 15px;
        background-color: #f8d7da;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .warning-box {
        padding: 15px;
        background-color: #fff3cd;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        padding: 10px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.9em;
        border-top: 1px solid #ddd;
        margin-top: 30px;
    }
    
    /* Model info card */
    .model-info {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    /* Prediction highlight */
    .prediction-result {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    
    /* Confidence bar */
    .confidence-text {
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# KHỞI TẠO SESSION STATE
# =============================================================================
def init_session_state():
    """Khởi tạo các biến trong session state"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_path' not in st.session_state:
        st.session_state.model_path = None
    if 'device' not in st.session_state:
        st.session_state.device = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'inference_time' not in st.session_state:
        st.session_state.inference_time = None
    if 'raw_logits' not in st.session_state:
        st.session_state.raw_logits = None
    if 'all_probs' not in st.session_state:
        st.session_state.all_probs = None
    if 'text_input_value' not in st.session_state:
        st.session_state.text_input_value = ""
    if 'max_length' not in st.session_state:
        st.session_state.max_length = config.MAX_SEQUENCE_LENGTH


# =============================================================================
# AUTO LOAD MODEL
# =============================================================================
def auto_load_model():
    """Tự động tải model đầu tiên tìm thấy"""
    model_files = get_model_files(config.MODEL_DIR)
    
    if not model_files:
        return None, None, None
    
    # Lấy model đầu tiên
    selected_model_path = model_files[0]
    selected_model_name = os.path.basename(selected_model_path)
    device = get_device()
    
    try:
        model = load_model(
            selected_model_path,
            device=device,
            num_labels=config.NUM_LABELS,
            architecture=config.MODEL_ARCHITECTURE
        )
        return model, selected_model_path, device
    except Exception as e:
        print(f"Lỗi auto-load model: {str(e)}")
        return None, None, None


# =============================================================================
# SIDEBAR - THÔNG TIN MODEL
# =============================================================================
def render_sidebar():
    """Hiển thị sidebar với thông tin model"""
    with st.sidebar:
        st.markdown("## ⚙️ Cấu hình Model")
        
        # Device info
        device = get_device()
        st.markdown(f"""
        <div class="model-info">
            <strong>🖥️ Thiết bị:</strong> {device}<br>
            <strong>🔧 CUDA Available:</strong> {"Có" if torch.cuda.is_available() else "Không"}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model files
        st.markdown("### 📁 File Model")
        
        model_files = get_model_files(config.MODEL_DIR)
        
        if not model_files:
            st.warning("⚠️ Không tìm thấy file model nào!")
            return None, None, None
        
        # Hiển thị model đang dùng
        if st.session_state.model_loaded and st.session_state.model_path:
            model_name = os.path.basename(st.session_state.model_path)
            file_size = get_model_file_size(st.session_state.model_path)
            st.markdown(f"""
            <div class="model-info">
                <strong>📦 Model:</strong> {model_name}<br>
                <strong>Kích thước:</strong> {file_size}
            </div>
            """, unsafe_allow_html=True)
            
            # Nút reload
            if st.button("🔄 Tải lại Model", use_container_width=True):
                st.rerun()
        
        # Model loaded status
        if st.session_state.model_loaded:
            st.markdown("""
            <div class="success-box">
                ✅ Model đã sẵn sàng
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Cài đặt")
        
        max_length = st.slider(
            "Độ dài tối đa:",
            min_value=64,
            max_value=512,
            value=config.MAX_SEQUENCE_LENGTH,
            step=64,
            key="max_length_slider"
        )
        
        architecture = st.selectbox(
            "Kiến trúc model:",
            ["feedforward", "transformer", "lstm", "cnn"],
            index=0,
            key="architecture_select"
        )
        
        return "", config.MAX_SEQUENCE_LENGTH, config.MODEL_ARCHITECTURE


# =============================================================================
# HEADER BANNER
# =============================================================================
def render_header():
    """Hiển thị header banner"""
    st.markdown(f"""
    <div class="header-banner">
        <h1>{config.APP_ICON} {config.APP_TITLE}</h1>
        <p>Ứng dụng Text Classification với PyTorch</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TEXT INPUT SECTION
# =============================================================================
def render_text_input(max_length: int):
    """
    Hiển thị phần nhập văn bản
    
    Returns:
        Tuple (text_input, predict_clicked, clear_clicked)
    """
    st.markdown('<p class="section-header">📝 Nhập văn bản</p>', unsafe_allow_html=True)
    
    # Initialize text value in session state
    if 'text_input_value' not in st.session_state:
        st.session_state.text_input_value = ""
    
    # Text area - dùng value từ session state
    text_input = st.text_area(
        "Nhập văn bản cần phân loại:",
        value=st.session_state.text_input_value,
        key="text_input_area_widget",
        height=150,
        placeholder="Nhập văn bản tiếng Anh hoặc tiếng Việt...",
        max_chars=max_length * 4
    )
    
    # Update session state với giá trị từ text area
    st.session_state.text_input_value = text_input
    
    # Buttons row - chỉ 2 nút
    col1, col2 = st.columns(2)
    
    with col1:
        predict_clicked = st.button("🔮 Dự đoán", type="primary", use_container_width=True)
    
    with col2:
        clear_clicked = st.button("🗑️ Xóa", use_container_width=True)
    
    return text_input, predict_clicked, clear_clicked


# =============================================================================
# PREDICTION OUTPUT SECTION
# =============================================================================
def render_prediction_output(prediction_result: str, confidence: float, 
                              all_probs: Dict[str, float], inference_time: float,
                              raw_logits: torch.Tensor):
    """Hiển thị kết quả dự đoán"""
    
    # Main prediction result
    st.markdown('<p class="section-header">📊 Kết quả dự đoán</p>', unsafe_allow_html=True)
    
    # Prediction card - dynamic styling
    pred_class = "prediction-positive"
    st.markdown(f"""
    <div class="prediction-result {pred_class}">
        🎯 Dự đoán: {prediction_result.upper()}<br>
        <span class="confidence-text">Độ tin cậy: {confidence*100:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")  # Spacing
    
    # Bar chart for all probabilities
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Xác suất các lớp")
        
        # Prepare data for bar chart
        import pandas as pd
        prob_df = pd.DataFrame({
            'Xác suất': all_probs
        })
        
        st.bar_chart(prob_df)
    
    with col2:
        # Inference time
        st.markdown("### ⏱️ Thời gian inference")
        st.metric("Thời gian", f"{inference_time:.2f} ms")
    
    # Raw logits in expandable section
    with st.expander("🔍 Debug: Xem raw logits"):
        st.write("**Logits (trước softmax):**")
        st.code(str(raw_logits[0].tolist()))
        
        st.write("**Probabilities (sau softmax):**")
        for label, prob in all_probs.items():
            st.write(f"  - {label}: {prob:.6f}")


# =============================================================================
# ERROR HANDLING
# =============================================================================
def handle_error(error: Exception, context: str = ""):
    """Xử lý và hiển thị lỗi"""
    error_msg = str(error)
    if context:
        error_msg = f"{context}: {error_msg}"
    
    st.markdown(f"""
    <div class="error-box">
        <strong>❌ Đã xảy ra lỗi!</strong><br>
        {error_msg}
    </div>
    """, unsafe_allow_html=True)
    
    # Show traceback in expander
    with st.expander("🔍 Xem chi tiết lỗi"):
        st.code(traceback.format_exc())


# =============================================================================
# FOOTER
# =============================================================================
def render_footer():
    """Hiển thị footer"""
    st.markdown(f"""
    <div class="footer">
        <p>© 2024 {config.APP_TITLE} | Phiên bản {config.VERSION}</p>
        <p>Powered by Streamlit + PyTorch</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Hàm chính của ứng dụng"""
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # Auto-load model on startup (chỉ chạy một lần)
    if not st.session_state.model_loaded:
        with st.spinner("Đang tự động tải model..."):
            model, model_path, device = auto_load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.model_path = model_path
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.toast("✅ Model đã tự động tải thành công!", icon="🎉")
    
    # Render header
    render_header()
    
    # Render sidebar and get settings
    model_path, max_length, architecture = render_sidebar()
    
    st.markdown("---")
    
    # Initialize demo text in session state if not exists
    # Render text input section - 2 nút
    text_input, predict_clicked, clear_clicked = render_text_input(max_length)
    
    # Handle clear button
    if clear_clicked:
        st.session_state.prediction_result = None
        st.session_state.inference_time = None
        st.session_state.raw_logits = None
        st.session_state.all_probs = None
        # Set text_input_value về rỗng
        st.session_state.text_input_value = ""
        st.rerun()
    
    # Get text input value from session state
    text_input = st.session_state.get('text_input_value', '')
    
    # Handle predict button
    if predict_clicked:
        # Check if model is loaded
        if not st.session_state.model_loaded or st.session_state.model is None:
            st.markdown("""
            <div class="warning-box">
                ⚠️ Vui lòng tải model trước khi dự đoán!
            </div>
            """, unsafe_allow_html=True)
        
        # Check if text is empty
        elif not text_input or text_input.strip() == "":
            st.markdown("""
            <div class="warning-box">
                ⚠️ Vui lòng nhập văn bản để dự đoán!
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Perform prediction
            try:
                with st.spinner("Đang dự đoán..."):
                    # Get model and device
                    model = st.session_state.model
                    device = st.session_state.device
                    
                    # Run prediction
                    logits, inference_time = predict(
                        model=model,
                        text=text_input,
                        device=device,
                        max_length=max_length
                    )
                    
                    # Get predictions
                    prediction, confidence = get_top_prediction(logits, config.LABEL_NAMES)
                    all_probs = get_predictions(logits, config.LABEL_NAMES)
                    
                    # Save to session state
                    st.session_state.prediction_result = prediction
                    st.session_state.confidence = confidence
                    st.session_state.all_probs = all_probs
                    st.session_state.inference_time = inference_time
                    st.session_state.raw_logits = logits
                    
            except Exception as e:
                handle_error(e, "Lỗi khi dự đoán")
    
    # Display prediction results if available
    if st.session_state.prediction_result is not None:
        render_prediction_output(
            prediction_result=st.session_state.prediction_result,
            confidence=st.session_state.confidence,
            all_probs=st.session_state.all_probs,
            inference_time=st.session_state.inference_time,
            raw_logits=st.session_state.raw_logits
        )
    
    # Render footer
    render_footer()


# =============================================================================
# RUN APP
# =============================================================================
if __name__ == "__main__":
    main()
