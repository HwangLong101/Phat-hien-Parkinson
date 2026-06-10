# Phân loại bệnh Parkinson qua hình vẽ bằng học sâu

Repository này chứa mã nguồn, dữ liệu và báo cáo cho đề tài phân loại bệnh Parkinson từ ảnh vẽ xoắn ốc/sóng bằng các mô hình học sâu sử dụng transfer learning.

## 1. Mục tiêu đề tài

Mục tiêu của đề tài là xây dựng hệ thống phân loại ảnh vẽ thành hai lớp:

- `healthy`: người khỏe mạnh
- `parkinson`: bệnh nhân Parkinson

Nhóm sử dụng các mô hình CNN pre-trained trên ImageNet, sau đó thêm phần head phân loại tùy chỉnh để phù hợp với bộ dữ liệu nhỏ.

## 2. Cấu trúc thư mục hiện tại

Repository hiện được sắp xếp như sau:

```text
.
├── dataset/
│   └── drawings/
│       ├── spiral/
│       │   ├── training/
│       │   │   ├── healthy/
│       │   │   └── parkinson/
│       │   └── testing/
│       │       ├── healthy/
│       │       └── parkinson/
│       └── wave/
│           ├── training/
│           │   ├── healthy/
│           │   └── parkinson/
│           └── testing/
│               ├── healthy/
│               └── parkinson/
│
├── main_training_notebook/
│   └── Parkinson_Demo_Paper_Custom_Dropout.ipynb
│
├── extended_experiment/
│   └── Parkinson_Demo_EfficientNetB0.ipynb
│
├── old_training_version/
│   ├── Parkinson_Demo_Optimized.ipynb
│   ├── Parkinson_Demo_Paper_Reproduction.ipynb
│   ├── Parkinson_Demo_For_Teacher (1).ipynb
│   └── Parkinsons.ipynb
│
├── report/
│   └── Báo cáo PBL4.docx
│
└── README.md
```

Trong đó:

- `dataset/drawings/`: chứa dữ liệu ảnh vẽ theo hai dạng `spiral` và `wave`.
- `dataset/drawings/spiral/`: dữ liệu ảnh xoắn ốc, được các notebook chính sử dụng.
- `dataset/drawings/wave/`: dữ liệu ảnh vẽ sóng, giữ kèm theo bộ dữ liệu gốc để tham khảo/thử nghiệm thêm.
- Mỗi dạng dữ liệu gồm hai split `training/`, `testing/` và hai lớp `healthy/`, `parkinson/`.
- `main_training_notebook/`: chứa notebook chính dùng để huấn luyện và đánh giá các mô hình trong báo cáo.
- `extended_experiment/`: chứa thử nghiệm mở rộng với EfficientNetB0.
- `old_training_version/`: chứa các bản notebook cũ/từng thử nghiệm trước đó để tham khảo quá trình phát triển.
- `report/`: chứa file báo cáo Word của đề tài.

## 3. Dataset

Bộ dữ liệu trong repository nằm tại:

```text
dataset/drawings/
```

Trong đó có hai loại ảnh:

- `dataset/drawings/spiral/`: ảnh vẽ xoắn ốc.
- `dataset/drawings/wave/`: ảnh vẽ sóng.

Thực nghiệm chính trong notebook sử dụng dữ liệu `spiral` gồm hai nhóm đối tượng:

- 51 ảnh thuộc lớp `healthy`
- 51 ảnh thuộc lớp `parkinson`

Tổng cộng tập `spiral` có 102 ảnh.

Trong thực nghiệm chính, dữ liệu được chia thành:

- 65 ảnh train
- 17 ảnh validation
- 20 ảnh test

Tập test được giữ cân bằng với 10 ảnh `healthy` và 10 ảnh `parkinson`.

## 4. Mô hình sử dụng

Thực nghiệm chính đánh giá bốn mô hình transfer learning:

- VGG19
- InceptionV3
- ResNet50V2
- DenseNet169

Các backbone sử dụng trọng số ImageNet, loại bỏ classifier gốc và thêm head phân loại tùy chỉnh.

Kiến trúc head chính:

```text
Backbone ImageNet
→ Global Average Pooling
→ Dense
→ Dropout(0.30)
→ Dense(2, Softmax)
```

Điểm cải tiến của nhóm là thêm lớp `Dropout(0.30)` sau Dense layer nhằm giảm overfitting trên bộ dữ liệu nhỏ.

## 5. Thử nghiệm mở rộng

Ngoài bốn mô hình chính, nhóm thử nghiệm thêm EfficientNetB0 như một mô hình mở rộng/tham khảo.

EfficientNetB0 sử dụng cấu trúc:

```text
EfficientNetB0 backbone
→ GlobalAveragePooling2D
→ Dense(128, ReLU)
→ Dropout(0.40)
→ Dense(1, Sigmoid)
```

Kết quả của EfficientNetB0 không dùng để so sánh trực tiếp tuyệt đối với bốn mô hình chính vì thiết lập chia dữ liệu khác.

## 6. Kết quả thực nghiệm chính

| Model | Accuracy bài báo gốc | Test Accuracy | Test AUC | Threshold | Best Epoch |
|---|---:|---:|---:|---:|---:|
| VGG19 | 72% | 95.00% | 0.9900 | 0.48 | 38 |
| InceptionV3 | 89% | 80.00% | 0.9200 | 0.50 | 43 |
| ResNet50V2 | 80% | 65.00% | 0.7600 | 0.46 | 1 |
| DenseNet169 | 85% | 80.00% | 0.9600 | 0.50 | 8 |

Mô hình tốt nhất trong thực nghiệm chính là:

```text
VGG19 + Dropout
```

Kết quả của VGG19 + Dropout:

- Test Accuracy: 95.00%
- Test AUC: 0.9900
- Recall lớp Parkinson: 1.0000

## 7. Kết quả EfficientNetB0 mở rộng

EfficientNetB0 được dùng như mô hình mở rộng/tham khảo.

Kết quả:

- Test Accuracy: 76.67%
- Test AUC: 0.8444
- Threshold: 0.34
- Parkinson recall: 0.7333

## 8. Ghi chú về so sánh với bài báo gốc

Kết quả thực nghiệm có thể khác với bài báo gốc do:

- Bài báo không công bố đầy đủ seed ngẫu nhiên.
- Bài báo không cung cấp đầy đủ mã nguồn huấn luyện.
- Cách chia train/validation/test có thể khác nhau.
- Bộ dữ liệu nhỏ, nên chỉ một vài ảnh đúng/sai cũng làm accuracy thay đổi đáng kể.

Vì vậy, bảng so sánh với bài báo gốc chỉ được dùng để tham khảo xu hướng hiệu quả giữa các mô hình, không nhằm tái lập tuyệt đối kết quả của bài báo.

## 9. Cách chạy

Khuyến nghị chạy notebook bằng Google Colab.

Các bước thực hiện:

1. Upload repository hoặc notebook lên Google Colab.
2. Chuẩn bị dữ liệu theo cấu trúc `dataset/drawings/spiral/{training,testing}/{healthy,parkinson}`.
3. Mở notebook chính trong thư mục `main_training_notebook/`.
4. Nếu chạy trên Colab, upload thư mục `dataset/drawings` lên Google Drive và chỉnh biến `DATASET_PATH` trong notebook tới thư mục `drawings/spiral` tương ứng.
5. Nếu chạy trực tiếp từ repository này, dùng đường dẫn dữ liệu `dataset/drawings/spiral`.
6. Chạy lần lượt các cell từ đầu đến cuối.
7. Xem kết quả ở các bảng tổng hợp, confusion matrix và ROC/AUC plot.

Notebook chính:

```text
main_training_notebook/Parkinson_Demo_Paper_Custom_Dropout.ipynb
```

Notebook thử nghiệm mở rộng:

```text
extended_experiment/Parkinson_Demo_EfficientNetB0.ipynb
```

## 10. Thư viện sử dụng

Các thư viện chính:

- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 11. File báo cáo

File báo cáo chi tiết nằm trong thư mục:

```text
report/Báo cáo PBL4.docx
```

## 12. Kết luận

Kết quả cho thấy transfer learning phù hợp với bài toán phân loại bệnh Parkinson từ hình vẽ xoắn ốc trên bộ dữ liệu nhỏ. Trong các mô hình chính, VGG19 + Dropout đạt kết quả tốt nhất với accuracy 95.00%, AUC 0.9900 và recall lớp Parkinson 1.0000.
