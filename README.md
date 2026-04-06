# Đánh giá và so sánh trong bài toán phân đoạn đối tượng trong video: CUTIE và XMem

<p align="center">
  Khung đánh giá so sánh cho bài toán phân đoạn đối tượng trong video sử dụng <b>CUTIE</b> và <b>XMem</b>
</p>

---

## Tổng quan

Dự án này phục vụ cho nghiên cứu về:

**Ứng dụng Deep Learning cho bài toán phân đoạn đối tượng trong video**

Mục tiêu chính là đánh giá và so sánh hai mô hình phân đoạn đối tượng trong video hiện đại:

- **CUTIE**
- **XMem**

Dự án bao gồm:
- quản lý kết quả suy luận
- nhiều độ đo đánh giá khác nhau
- trực quan hóa định tính bằng video chồng lớp kết quả

---

## Cấu trúc dự án

```bash
.
├── assets/
│   ├── Efficiency/
│   ├── Object Persistence/
│   ├── Results/
│   ├── Shape Quality/
│   └── Temporal Stability/
│
├── data/
│   ├── Annotations/
│   └── JPEGImages/
│
├── evaluation/
│   ├── efficiency.py
│   ├── object_persistence.py
│   ├── shape_quality.py
│   ├── temporal_stability.py
│   └── video_demo.py
│
├── experiments/
│   ├── results_cutie/
│   └── results_xmem/
│
├── model/
│   ├── Cutie/
│   └── XMem/
│
└── scripts/
    ├── prepare_dataset.sh
    ├── run_cutie.sh
    └── run_xmem.sh
```

---

## Mô tả các thư mục

### `assets/`
Lưu trữ các tài nguyên đầu ra dùng cho trực quan hóa, báo cáo và thuyết trình.

Các thư mục con:
- `Efficiency/` — hình ảnh và kết quả đánh giá hiệu năng
- `Object Persistence/` — kết quả đánh giá khả năng duy trì đối tượng
- `Results/` — các đầu ra tổng quát và trực quan hóa cuối cùng
- `Shape Quality/` — kết quả đánh giá chất lượng hình dạng mask
- `Temporal Stability/` — kết quả đánh giá độ ổn định theo thời gian

---

### `data/`
Chứa bộ dữ liệu dùng để đánh giá.

- `JPEGImages/` — các khung hình RGB của từng chuỗi
- `Annotations/` — các mask phân đoạn ground truth

Ví dụ:
```bash
data/
├── Annotations/
│   ├── 0ql59q5s/
│   ├── 1wjebgyd/
│   └── 3v2kgn6k/
└── JPEGImages/
    ├── 0ql59q5s/
    ├── 1wjebgyd/
    └── 3v2kgn6k/
```

---

### `evaluation/`
Chứa các script dùng để tính toán độ đo và so sánh định tính.

- `efficiency.py` — đánh giá hiệu năng thời gian chạy của CUTIE và XMem
- `object_persistence.py` — đánh giá mức độ duy trì đối tượng qua các khung hình
- `shape_quality.py` — đánh giá chất lượng mask dự đoán
- `temporal_stability.py` — đánh giá độ ổn định của phân đoạn theo thời gian
- `video_demo.py` — tạo video so sánh dạng overlay giữa CUTIE và XMem

---

### `experiments/`
Chứa kết quả suy luận của từng mô hình.

- `results_cutie/` — các mask dự đoán từ CUTIE
- `results_xmem/` — các mask dự đoán từ XMem

Ví dụ:
```bash
experiments/
├── results_cutie/
│   ├── 0ql59q5s/
│   ├── 1wjebgyd/
│   └── 3v2kgn6k/
└── results_xmem/
    ├── 0ql59q5s/
    ├── 1wjebgyd/
    └── 3v2kgn6k/
```

---

### `model/`
Chứa mã nguồn hoặc thư mục mô hình cục bộ của hai phương pháp VOS.

- `Cutie/`
- `XMem/`

Thư mục này được sử dụng để:
- chạy suy luận
- tổ chức checkpoint
- tham chiếu đến các triển khai gốc của mô hình

---

### `scripts/`
Chứa các script shell để chạy toàn bộ quy trình.

- `prepare_dataset.sh` — chuẩn bị bộ dữ liệu
- `run_cutie.sh` — chạy suy luận với CUTIE
- `run_xmem.sh` — chạy suy luận với XMem

---

## Quy trình làm việc

### 1. Chuẩn bị bộ dữ liệu
Đặt bộ dữ liệu theo cấu trúc sau:

```bash
data/JPEGImages/<ten_chuoi_du_lieu>/
data/Annotations/<ten_chuoi_du_lieu>/
```

Sau đó chạy:

```bash
bash scripts/prepare_dataset.sh
```

---

### 2. Chạy suy luận với CUTIE

```bash
bash scripts/run_cutie.sh
```

Kết quả đầu ra sẽ được lưu tại:

```bash
experiments/results_cutie/
```

---

### 3. Chạy suy luận với XMem

```bash
bash scripts/run_xmem.sh
```

Kết quả đầu ra sẽ được lưu tại:

```bash
experiments/results_xmem/
```

---

### 4. Chạy đánh giá

```bash
python evaluation/efficiency.py
python evaluation/object_persistence.py
python evaluation/shape_quality.py
python evaluation/temporal_stability.py
```

---

### 5. Tạo video so sánh định tính

```bash
python evaluation/video_demo.py
```

Các video được tạo ra và đầu ra trực quan thường được lưu trong:

```bash
assets/Results/
```

---

## Các hạng mục đánh giá

### 1. Hiệu năng
Đánh giá các chỉ số liên quan đến thời gian chạy:
- thời gian xử lý trên mỗi khung hình
- độ trễ trung bình
- FPS trung bình
- so sánh độ ổn định độ trễ giữa CUTIE và XMem

### 2. Khả năng duy trì đối tượng
Đánh giá mức độ mô hình giữ được đối tượng xuyên suốt video:
- tính liên tục của đối tượng
- khả năng duy trì đối tượng theo thời gian
- độ bền vững khi đối tượng có nguy cơ bị mất dấu

### 3. Chất lượng hình dạng
Đánh giá chất lượng của mask phân đoạn dự đoán:
- mức độ đầy đủ của vùng đối tượng
- chất lượng đường biên
- độ tương đồng với mask ground truth

### 4. Độ ổn định theo thời gian
Đánh giá sự nhất quán giữa các khung hình liên tiếp:
- giảm hiện tượng nhấp nháy
- giảm rung lắc của mask
- duy trì phân đoạn ổn định theo thời gian

---

## Các chuỗi dữ liệu ví dụ

Một số chuỗi tiêu biểu được dùng trong quá trình đánh giá:

- `0ql59q5s`
- `1wjebgyd`
- `3v2kgn6k`

---

## Hướng phát triển trong tương lai

Một số hướng mở rộng cho dự án:
- Nghiên cứu thay thế kiến trúc Transformer trong CUTIE bằng Mamba (State Space Model - SSM) để đánh giá tiềm năng cải thiện hiệu năng và giảm chi phí tính toán
- Tối ưu mô hình và toàn bộ pipeline xử lý để đạt suy luận thời gian thực, từ đó nâng cao tính thực tiễn trong các ứng dụng thực tế
