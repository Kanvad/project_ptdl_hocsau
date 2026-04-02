# PubMed RCT Classification

Dự án phân loại văn bản y khoa từ tập dữ liệu PubMed 200k RCT.

## Mô tả

Dự án sử dụng mô hình SciBERT để phân loại câu trong tóm tắt y khoa (abstract) vào 5 nhãn:
- BACKGROUND (0)
- OBJECTIVE (1)
- METHODS (2)
- RESULTS (3)
- CONCLUSIONS (4)

## Cấu trúc thư mục

```
project_ptdl_hs/
├── data/                    # Dữ liệu
│   ├── train.csv           # Dữ liệu huấn luyện
│   ├── val.csv             # Dữ liệu validation
│   ├── test.csv            # Dữ liệu kiểm tra
│   └── PubMed_200k_RCT/    # Dữ liệu gốc
├── models/                  # Mô hình đã huấn luyện
├── notebooks/               # Jupyter notebooks (EDA, training)
├── results/                 # Kết quả visualization
└── preprocess.py           # Script tiền xử lý dữ liệu
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### Tiền xử lý dữ liệu

```bash
python preprocess.py --input data/PubMed_200k_RCT/train.txt --output data
```

### Huấn luyện

Chạy notebook `notebooks/notebook_train_optimized.ipynb` để huấn luyện mô hình.

## Yêu cầu

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas, numpy
