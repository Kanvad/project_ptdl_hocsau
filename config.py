"""
Cấu hình cho ứng dụng Streamlit Text Classification
Chứa các thiết lập về nhãn, độ dài tối đa, và kiến trúc model
"""

# Danh sách tên nhãn cho classification (PubMed RCT 5 classes)
LABEL_NAMES = [
    "BACKGROUND",
    "OBJECTIVE", 
    "METHODS",
    "RESULTS",
    "CONCLUSIONS"
]

# Số lượng nhãn (độ dài của LABEL_NAMES)
NUM_LABELS = 5

# Độ dài tối đa của chuỗi văn bản đầu vào (tính theo token)
MAX_SEQUENCE_LENGTH = 512  # Phù hợp cho BERT/SciBERT

# Kiểu kiến trúc model (PubMed sử dụng SciBERT)
MODEL_ARCHITECTURE = "transformer"  # Hoặc "feedforward", "lstm", "cnn"

# Đường dẫn thư mục chứa model
MODEL_DIR = "models"

# Các loại file model được hỗ trợ
SUPPORTED_MODEL_EXTENSIONS = [".pth", ".pt"]

# Cổng mặc định cho Streamlit
STREAMLIT_PORT = 8501

# Tiêu đề ứng dụng
APP_TITLE = "PubMed RCT Classification"
APP_ICON = "🏥"

# Mẫu văn bản để demo (PubMed RCT abstract mẫu)
DEMO_TEXT = "BACKGROUND: Recent studies have shown that machine learning algorithms can improve diagnostic accuracy in medical imaging. OBJECTIVE: The aim of this study was to evaluate the effectiveness of deep learning in detecting early-stage tumors. METHODS: We trained a convolutional neural network on 10,000 CT scans and validated on 2,000 images. RESULTS: The model achieved 95% accuracy with a sensitivity of 93% and specificity of 97%. CONCLUSIONS: Deep learning shows promise for early tumor detection in clinical practice."

# Thông tin phiên bản
VERSION = "1.0.0"
