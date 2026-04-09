# Kế hoạch triển khai mô hình Hierarchical Bi-LSTM + CRF

## Phase A: Tiền xử lý dữ liệu (50% độ chính xác)

### 1. Chuẩn bị dữ liệu
- Tải dataset từ GitHub của tác giả

### 2. Tokenization (Tách từ)
- **Khuyến nghị:** spaCy với model `en_core_sci_sm` (chuyên y khoa, tối ưu cho Linux)
pip install scispacy
- Thay thế tất cả chữ số bằng ký tự `@` (giảm vocabulary size, tập trung cấu trúc ngữ pháp)

### 3. Xây dựng Vocabulary & Padding
- Lọc bỏ từ xuất hiện < 3 lần
- Pad câu về `max_words`, pad abstract về `max_sentences`
- **Lưu ý:** Giữ lại Mask (vị trí thật vs padding) cho tầng CRF

---

## Phase B: Kiến trúc mô hình

### Tầng 1: Word Embeddings
- **Bắt buộc:** Dùng pre-trained vectors (BioWordVec hoặc PubMed GloVe)
- Không dùng Random Initialization → boost F1-score vài %

### Tầng 2: Token-level Bi-LSTM
- Đọc từng từ trong câu
- Sử dụng `pack_padded_sequence` của PyTorch để bỏ qua padding
- Output: vector đại diện cho mỗi câu

### Tầng 3: Sentence-level Bi-LSTM
- Đọc chuỗi vector câu từ Tầng 2
- Học ngữ cảnh (VD: "Objective → Methods")

### Tầng 4: CRF Layer
- Tính toán đường đi (path) nhãn hợp lý nhất cho toàn bộ abstract

---

## Phase C: Chiến lược huấn luyện

### Loss Function
- Negative Log-Likelihood từ CRF layer

### Optimizer
- **AdamW** với LR: 1e-3 hoặc 3e-4
- **LRScheduler:** ReduceLROnPlateau (giảm LR khi loss đi ngang)

### Regularization
| Kỹ thuật | Giá trị |
|----------|---------|
| Dropout | 0.4 - 0.5 (giữa các LSTM, trước Linear) |
| Early Stopping | Dừng sau 5 epochs nếu F1 không tăng |

---

## Phase D: Đánh giá & Fine-tuning

### Công cụ đánh giá
- Classification Report
- Confusion Matrix

### Vấn đề thường gặp
- **Background vs Objective:** 757 câu Objective bị nhầm thành Background

### Cách khắc phục
- Tăng `class_weight` cho class Objective
- Thêm feature thủ công (vị trí câu trong đoạn văn) trước tầng CRF

---

## Phase E: Triển khai

### Khi đạt F1 > 90%
1. Lưu model `.pth`
2. Xây API với Flask/FastAPI
   - Input: abstract
   - Output: nhãn từng câu
3. Đưa preprocessing (tokenization, padding) vào pipeline API
