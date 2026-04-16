# 🍎 Hệ Thống Phân Loại & Kiểm Tra Chất Lượng Trái Cây

**Fruit Quality Inspection & Classification System**

> 📚 Môn học: Xử lý Ảnh và Thị Giác Máy Tính (121036)  
> 🏫 Trường: Đại học Giao thông Vận tải TP. Hồ Chí Minh  
> ⚖️ Trọng số: 50% điểm tổng kết

---

## 🎯 Mô tả dự án

Hệ thống tự động **phân loại loại trái cây** và **đánh giá chất lượng** bằng các kỹ thuật xử lý ảnh và thị giác máy tính. Kết quả đầu ra là nhãn kết hợp dạng:

```
Apple_Good  |  Banana_Bad  |  Orange_Mixed  |  Guava_Good  |  ...
```

### Dataset
- **Nguồn:** [Kaggle — Fruit Quality Classification](https://www.kaggle.com)
- **Loại trái cây:** Apple · Banana · Guava · Lime · Orange
- **Chất lượng:** Good (tươi) · Bad (hỏng) · Mixed (trung bình)
- **Tổng số lớp:** 15 lớp (5 loại × 3 chất lượng)

---

## 🔬 Kỹ thuật áp dụng

| Chương | Kỹ thuật | Chi tiết |
|--------|----------|----------|
| **Ch.2** | Gaussian filter | Khử nhiễu điều kiện chụp kém |
| **Ch.2** | Median filter | Khử nhiễu salt-and-pepper, bảo toàn cạnh |
| **Ch.2** | Bilateral filter | Khử nhiễu + giữ cạnh sắc |
| **Ch.2** | CLAHE | Cân bằng histogram cục bộ trên kênh L-LAB |
| **Ch.2** | Chuyển HSV / LAB | Phân tích màu sắc độc lập với ánh sáng |
| **Ch.3** | Canny Edge Detection | Phát hiện vết nứt, rãnh trên bề mặt |
| **Ch.3** | Sobel Gradient | Cường độ biến đổi pixel — vết thâm |
| **Ch.3** | HOG | Mô tả hình dạng tổng thể (~1764 chiều) |
| **Ch.3** | LBP | Mô tả texture bề mặt (64 chiều) |
| **Ch.3** | Histogram màu HSV | Đặc trưng màu sắc và độ chín (96 chiều) |
| **Ch.4** | K-Means Clustering | Phân cụm pixel — tách nền theo màu |
| **Ch.4** | GrabCut | Phân đoạn chính xác (GMM + Graph Cut) |
| **Ch.4** | Watershed | Tách vùng khuyết tật dính nhau |
| **Ch.4** | HSV Thresholding | Phát hiện vùng thâm/đen/hỏng |
| **Ch.5** | SVM (kernel RBF) | Phân loại 15 lớp từ vector đặc trưng |

> ✅ **Đáp ứng yêu cầu:** ≥ 3 kỹ thuật từ ≥ 3 chương (Ch.2, Ch.3, Ch.4, Ch.5)

---

## 📁 Cấu trúc dự án

```
fruit_quality_project/
│
├── 📓 01_Dataset_Preparation.ipynb   ← Notebook 1: Tổ chức lại dataset
├── 📓 02_FruitQuality_Pipeline.ipynb ← Notebook 2: Pipeline hoàn chỉnh
├── 📄 README.md
│
├── data/                             ← Dataset Kaggle gốc (giữ nguyên)
│   ├── Bad Quality_Fruits/
│   │   ├── Apple_Bad/
│   │   ├── Banana_Bad/ ...
│   ├── Good Quality_Fruits/
│   │   ├── Apple_Good/ ...
│   └── Mixed Qualit_Fruits/
│       ├── Apple_Mixed/ ...
│
├── dataset/                          ← Được tạo bởi Notebook 1
│   ├── train/      (70%)
│   │   ├── Apple_Bad/
│   │   ├── Apple_Good/
│   │   ├── Apple_Mixed/
│   │   ├── Banana_Bad/ ...
│   ├── valid/      (15%)
│   │   └── ...
│   └── test/       (15%)
│       └── ...
│
├── models/                           ← Model đã huấn luyện
│   └── fruit_svm.pkl
│
└── results/                          ← Biểu đồ và ảnh kết quả
    ├── 01_dataset_distribution.png
    ├── 01_sample_images.png
    ├── 01_split_distribution.png
    ├── 02_test_samples.png
    ├── 02_filter_comparison.png
    ├── 02_color_spaces.png
    ├── 03_segmentation_demo.png
    ├── 04_feature_extraction.png
    ├── 06_confusion_matrix.png
    ├── 06_roc_curves.png
    ├── 06_metrics_summary.png
    ├── 06_prediction_samples.png
    ├── 07_inference_all_classes.png
    └── 07_single_inference.png
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- Python **3.9+**
- RAM: ≥ 8 GB
- OS: Windows / Linux / macOS

### Cài đặt thư viện

```bash
pip install opencv-python scikit-image scikit-learn numpy matplotlib seaborn tqdm joblib pillow
```

Hoặc dùng file requirements:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python>=4.9.0
scikit-image>=0.22.0
scikit-learn>=1.4.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
joblib>=1.4.0
Pillow>=10.3.0
```

---

## 🚀 Hướng dẫn chạy

### Bước 1 — Chuẩn bị dataset

Tải dataset từ Kaggle và đặt vào thư mục `data/`:

```
data/
├── Bad Quality_Fruits/
│   ├── Apple_Bad/     ← ảnh .jpg
│   ├── Banana_Bad/
│   ├── Guava_Bad/
│   ├── Lime_Bad/
│   └── Orange_Bad/
├── Good Quality_Fruits/
│   └── ...
└── Mixed Qualit_Fruits/
    └── ...
```

### Bước 2 — Chạy Notebook 1: Tổ chức dataset

```bash
jupyter notebook 01_Dataset_Preparation.ipynb
```

1. Mở file, **chỉnh sửa `KAGGLE_ROOT`** trỏ đến thư mục `data/`
2. Chạy **Kernel → Restart & Run All**
3. Kết quả: thư mục `dataset/train/`, `dataset/valid/`, `dataset/test/`

### Bước 3 — Chạy Notebook 2: Pipeline hoàn chỉnh

```bash
jupyter notebook 02_FruitQuality_Pipeline.ipynb
```

Chạy **Kernel → Restart & Run All**. Pipeline sẽ:
1. Đọc dataset từ `dataset/`
2. Tiền xử lý on-the-fly (resize 224×224, CLAHE, ...)
3. Phân đoạn từng ảnh (GrabCut + Watershed)
4. Trích xuất đặc trưng (~1958 chiều)
5. Huấn luyện SVM (15 lớp)
6. Đánh giá và lưu kết quả

---

## 📊 Đặc trưng Vector (~1958 chiều)

```
[  HOG  |  LBP  | ColorHist | ColorStats | Shape | Edge | Defect ]
[ ~1764 |   64  |     96    |     18     |  12   |   3  |    1   ]
```

| Nhóm | Chiều | Ý nghĩa |
|------|-------|---------|
| HOG | ~1764 | Hình dạng tổng thể (hình cầu vs dài) |
| LBP | 64 | Texture bề mặt (nhám/mịn/nứt) |
| Color Hist HSV | 96 | Màu sắc và độ chín |
| Color Stats | 18 | Mean/Std của BGR, HSV, LAB |
| Shape | 12 | Circularity, solidity, Hu moments |
| Edge | 3 | Edge density, mean gradient, n_edges |
| Defect Ratio | 1 | Tỷ lệ diện tích khuyết tật |

---

## 🔍 Kết quả Inference

Mỗi ảnh đầu vào → hệ thống trả về:

```python
{
    'prediction':   'Apple_Good',    # Nhãn kết quả chính
    'fruit':        'Apple',         # Loại trái cây
    'quality':      'Good',          # Chất lượng
    'confidence':   0.923,           # Độ tin cậy (0–1)
    'defect_ratio': 0.024,           # Tỷ lệ khuyết tật (0–1)
    'top3': [                        # Top-3 dự đoán
        ('Apple_Good', 0.923),
        ('Apple_Mixed', 0.061),
        ('Guava_Good', 0.012),
    ],
    'processing_ms': 245.3           # Thời gian xử lý (ms)
}
```

---

## 👥 Phân công công việc

| Thành viên | Vai trò | Notebook / Section |
|------------|---------|-------------------|
| **TV1** (Nhóm trưởng) | Data Engineer + PM | Notebook 1 toàn bộ + README |
| **TV2** | Preprocessing (Ch.2) | Notebook 2 — Ô 2 |
| **TV3** | Segmentation (Ch.4) | Notebook 2 — Ô 3 |
| **TV4** | Feature Engineering (Ch.3) | Notebook 2 — Ô 4 |
| **TV5** | ML Engineer (Ch.5) | Notebook 2 — Ô 5 |
| **TV6** | Evaluation + Slides | Notebook 2 — Ô 6, 7 |

---

## 📚 Tài liệu tham khảo

1. Gonzalez, R.C. & Woods, R.E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR 2005*.
3. Ojala, T. et al. (2002). Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. *IEEE TPAMI, 24*(7).
4. Rother, C. et al. (2004). GrabCut: Interactive Foreground Extraction Using Iterated Graph Cuts. *ACM SIGGRAPH 2004*.
5. Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer.
6. Khojastehnazhand, M. et al. (2010). Development of a lemon sorting system based on color and size. *African Journal of Plant Science, 4*(4).
7. Moallem, P. et al. (2017). Computer vision-based apple grading for golden delicious apples. *Information Processing in Agriculture, 4*(1).

---

## 🐛 Troubleshooting

**Không tìm thấy ảnh (`scan_kaggle_dataset` trả về 0):**
- Kiểm tra `KAGGLE_ROOT` trỏ đúng thư mục chứa 3 folder `Bad/Good/Mixed`
- Đảm bảo ảnh nằm trực tiếp trong `Apple_Bad/`, `Banana_Good/`, ...

**GrabCut chậm:**
- Giảm số ảnh demo hoặc tắt `use_seg=False` trong `build_feature_matrix`

**Memory error khi trích xuất feature:**
- Giảm số ảnh train hoặc dùng `use_seg=False`

**SVM accuracy thấp:**
- Đảm bảo dataset đủ ảnh thực tế (không dùng ảnh giả lập)
- Thử bật GridSearchCV (Ô 5, bỏ comment phần GridSearch)
- Tăng `SVM_C` trong cấu hình

---

*© 2025–2026 — Môn Xử lý Ảnh và Thị Giác Máy Tính (121036) — ĐH GTVT TP.HCM*
