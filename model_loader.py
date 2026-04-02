"""
Model Loader Module - Hỗ trợ tải model PyTorch cho text classification
Hỗ trợ cả model lưu state_dict và full model
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
import glob


class DummyTextClassifier(nn.Module):
    """
    Model mẫu dạng Feedforward cho text classification
    (Dùng khi không tìm thấy model thực tế)
    """
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_labels: int = 5, 
                 dropout: float = 0.3):
        super(DummyTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_labels)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        pooled = embedded.mean(dim=1)  # (batch_size, embedding_dim)
        
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TransformerTextClassifier(nn.Module):
    """
    Model dạng Transformer cho text classification
    """
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, 
                 hidden_dim: int = 256, num_labels: int = 5,
                 max_seq_len: int = 512, dropout: float = 0.3):
        super(TransformerTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_labels)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embedded = self.embedding(x) + self.position_embedding(positions)
        
        encoded = self.transformer_encoder(embedded)
        pooled = encoded.mean(dim=1)
        
        output = self.dropout(pooled)
        output = self.fc(output)
        return output


def get_device() -> torch.device:
    """
    Tự động phát hiện và trả về device (CPU/CUDA)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model_files(model_dir: str) -> List[str]:
    """
    Quét thư mục model và trả về danh sách các file .pth/.pt
    
    Args:
        model_dir: Đường dẫn thư mục chứa model
        
    Returns:
        Danh sách đường dẫn tuyệt đối của các file model
    """
    model_files = []
    
    if not os.path.exists(model_dir):
        return model_files
    
    for ext in ["*.pth", "*.pt"]:
        pattern = os.path.join(model_dir, ext)
        files = glob.glob(pattern)
        model_files.extend(files)
    
    # Sắp xếp theo thời gian sửa đổi (mới nhất trước)
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return model_files


def get_model_file_size(model_path: str) -> str:
    """
    Lấy kích thước file model dưới dạng string
    
    Args:
        model_path: Đường dẫn file model
        
    Returns:
        Chuỗi kích thước file (VD: "15.2 MB")
    """
    size_bytes = os.path.getsize(model_path)
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def load_model_state_dict(model_path: str, device: torch.device, 
                          num_labels: int = 5) -> Tuple[nn.Module, bool]:
    """
    Tải model từ file .pth/.pt (hỗ trợ cả state_dict và full model)
    Hỗ trợ SciBERT và các model HuggingFace
    
    Args:
        model_path: Đường dẫn file model
        device: Thiết bị để load model (cpu/cuda)
        num_labels: Số lượng nhãn (chỉ dùng khi tạo model mới)
        
    Returns:
        Tuple gồm (model đã load, có phải full model hay không)
    """
    is_full_model = False
    
    try:
        # Thử load như full model trước (SciBERT/huggingface format)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Kiểm tra xem có phải là full model không
        # SciBERT/HuggingFace models thường là nn.Module hoặc có 'model_state_dict'
        if isinstance(checkpoint, nn.Module):
            model = checkpoint.to(device)
            model.eval()
            is_full_model = True
            return model, is_full_model
        elif isinstance(checkpoint, dict):
            # Kiểm tra xem có phải state_dict không
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Tạo model và load state_dict
            model = DummyTextClassifier(num_labels=num_labels)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            return model, is_full_model
        else:
            raise ValueError("Unknown checkpoint format")
            
    except Exception as e:
        # Nếu load thất bại, thử cách khác
        raise RuntimeError(f"Không thể tải model: {str(e)}")


def load_scibert_model(model_path: str, device: torch.device, num_labels: int = 5) -> nn.Module:
    """
    Tải SciBERT model cho text classification
    
    Args:
        model_path: Đường dẫn file model
        device: Thiết bị
        num_labels: Số lượng nhãn
        
    Returns:
        Model đã load
    """
    from transformers import BertForSequenceClassification, BertConfig
    
    # Load state_dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Xử lý prefix _orig_mod.
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Lấy cấu hình từ state_dict
    config = BertConfig()
    config.num_labels = num_labels
    config.vocab_size = new_state_dict.get('bert.embeddings.word_embeddings.weight').shape[0] if 'bert.embeddings.word_embeddings.weight' in new_state_dict else 28996
    config.hidden_size = new_state_dict.get('bert.embeddings.word_embeddings.weight').shape[-1] if 'bert.embeddings.word_embeddings.weight' in new_state_dict else 768
    config.num_hidden_layers = 12
    config.num_attention_heads = 12
    config.intermediate_size = 3072
    
    # Tạo model
    model = BertForSequenceClassification(config)
    
    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def load_model(model_path: str, device: Optional[torch.device] = None,
               num_labels: int = 5, architecture: str = "transformer") -> nn.Module:
    """
    Tải model từ file với kiến trúc được chỉ định
    Ưu tiên load model thực từ file
    
    Args:
        model_path: Đường dẫn file model
        device: Thiết bị (nếu None sẽ tự động phát hiện)
        num_labels: Số lượng nhãn
        architecture: Loại kiến trúc ("feedforward", "transformer", "lstm", "cnn")
        
    Returns:
        Model đã được load
    """
    if device is None:
        device = get_device()
    
    print(f"Đang tải model từ: {model_path}")
    
    try:
        # Thử load với SciBERT
        model = load_scibert_model(model_path, device, num_labels)
        print("Đã tải SciBERT model thành công!")
        return model
        
    except Exception as e:
        print(f"Lỗi load SciBERT: {str(e)}")
        
        try:
            # Fallback: load trực tiếp như full model
            model = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(model, nn.Module):
                model.eval()
                print("Đã tải full model!")
                return model
        except:
            pass
        
        # Tạo model mới
        print("Tạo model mới...")
        if architecture == "transformer":
            model = TransformerTextClassifier(num_labels=num_labels)
        else:
            model = DummyTextClassifier(num_labels=num_labels)
        
        model = model.to(device)
        model.eval()
        return model


def preprocess_text(text: str, max_length: int = 512) -> torch.Tensor:
    """
    Hàm tiền xử lý văn bản mẫu
    (Placeholder - cần thay thế bằng tokenizer thực tế)
    
    Args:
        text: Văn bản đầu vào
        max_length: Độ dài tối đa
        
    Returns:
        Tensor chứa indices của văn bản
    """
    # Tokenize đơn giản (placeholder)
    words = text.lower().split()[:max_length]
    
    # Tạo mapping đơn giản (trong thực tế nên dùng tokenizer có sẵn)
    vocab = {}
    for i, word in enumerate(set(words), start=1):
        vocab[word] = i
    
    # Chuyển đổi thành indices
    indices = [vocab.get(w, 0) for w in words]
    
    # Padding
    if len(indices) < max_length:
        indices.extend([0] * (max_length - len(indices)))
    
    return torch.tensor([indices], dtype=torch.long)


def predict(model: nn.Module, text: str, device: torch.device,
            max_length: int = 512) -> Tuple[torch.Tensor, float]:
    """
    Thực hiện inference trên văn bản
    
    Args:
        model: Model đã load
        text: Văn bản đầu vào
        device: Thiết bị để inference
        max_length: Độ dài tối đa
        
    Returns:
        Tuple gồm (logits, inference_time_ms)
    """
    import time
    
    # Thử dùng transformers tokenizer
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        end_time = time.time()
        
    except Exception as e:
        print(f"Lỗi tokenizer: {str(e)}")
        # Fallback: dùng placeholder
        input_tensor = preprocess_text(text, max_length).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            logits = model(input_tensor)
        end_time = time.time()
    
    inference_time_ms = (end_time - start_time) * 1000
    
    return logits, inference_time_ms


def get_predictions(logits: torch.Tensor, label_names: List[str]) -> Dict[str, float]:
    """
    Chuyển đổi logits thành probabilities và labels
    
    Args:
        logits: Tensor chứa logits từ model
        label_names: Danh sách tên nhãn
        
    Returns:
        Dictionary với nhãn và xác suất tương ứng
    """
    # Tính softmax
    probs = torch.softmax(logits, dim=-1)[0]
    
    # Tạo dictionary kết quả
    results = {}
    for i, label in enumerate(label_names):
        results[label] = probs[i].item()
    
    return results


def get_top_prediction(logits: torch.Tensor, label_names: List[str]) -> Tuple[str, float]:
    """
    Lấy nhãn có xác suất cao nhất
    
    Args:
        logits: Tensor chứa logits từ model
        label_names: Danh sách tên nhãn
        
    Returns:
        Tuple (nhãn, xác suất)
    """
    probs = torch.softmax(logits, dim=-1)[0]
    top_idx = int(probs.argmax(dim=-1).item())
    confidence = float(probs[top_idx].item())
    
    return label_names[top_idx], confidence
