# Model-5: MobileNetV3Large

Dokumentasi resmi untuk **Model-5 (`MobileNetV3Large`)** pada eksperimen perbandingan 5 arsitektur CNN untuk klasifikasi 10 kategori gambar pada dataset SIMPLIcity (1.000 citra), Tugas Kelompok Neuro Computing (Minggu 11), PENS.

---

## 1. Arsitektur & Ide Kunci Desain

* **Fungsi Keras:** `tf.keras.applications.MobileNetV3Large`
* **Bobot Awal:** Pretrained ImageNet (`weights="imagenet"`)
* **Input Resolusi:** $224 \times 224 \times 3$ (Letterbox padding dengan warna rata-rata ImageNet `(124, 116, 104)`)
* **Normalisasi:** `keras.applications.mobilenet_v3.preprocess_input` (rentang $[-1, 1]$ dipasang di dalam model)
* **Jumlah Parameter:** $\approx 4.2 \text{ juta}$ parameter ($\approx 2.5 \text{ juta}$ jika $\alpha = 0.75$)
* **Ide Kunci Arsitektur:**
  1. **Depthwise Separable Convolutions:** Memisahkan penyaringan spasial ($3 \times 3$ / $5 \times 5$ Depthwise) dan kombinasi representasi kanal ($1 \times 1$ Pointwise), memangkas beban komputasi FLOPs dan parameter hingga $\approx 85-90\%$ dibanding konvolusi standar.
  2. **Squeeze-and-Excitation (SE) Attention Block:** Modul kanal adaptif yang menimbang tingkat kepentingan fitur per-kanal melalui mekanisme Squeeze (*Global Average Pooling*) dan Excitation (*2 Dense Layers* dengan aktivasi non-linear).
  3. **Hard-Swish ($h\text{-swish}$):** Fungsi aktivasi non-linear efisien $h\text{-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$ yang mengaproksimasi Swish tanpa komputasi eksponensial mahal.

### Alasan Pemilihan & Kesesuaian dengan Dataset SIMPLIcity
1. **Skala Parameter Ideal untuk Dataset Kecil:** Dataset SIMPLIcity hanya memiliki 500 citra training (50 citra per kelas). Model raksasa (ResNet152 atau DenseNet201) sangat rentan mengalami *overfitting* parah. MobileNetV3Large yang ringkas ($\approx 4.2\text{M}$ parameter) menawarkan regularisasi kapasitas yang alami (*strong inductive bias*).
2. **Efisiensi & Latensi Rendah:** Dirancang untuk inferensi cepat pada perangkat komputasi terbatas dengan latensi rendah ($\approx 10\text{ ms}$ per citra).

---

## 2. Log 10 Trial Hyperparameter Tuning (`trials.csv`)

Seluruh 10 trial dijalankan pada GPU Kaggle (Tesla T4) dengan protokol fixed seed (`SEED = 42`), early stopping `patience=8`, dan batas maksimum 40 epoch:

| Trial | Alpha | Minimalistic | Pooling | Learning Rate | Optimizer | Unfreeze % | Batch Size | Dropout | Val Acc (%) | Val Loss | Best Epoch | Train Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.0 | False | avg | 0.001 | adam | 0% | 32 | 0.2 | 18.33% | 1,025,098.25 | 20 | 47.27 |
| **2** | 1.0 | False | avg | 0.0005 | adam | 0% | 32 | 0.3 | 20.00% | 970,760.75 | 35 | 58.35 |
| **3** | 0.75 | False | avg | 0.001 | adam | 0% | 32 | 0.2 | 16.00% | 521.99 | 38 | 54.34 |
| **4** | **1.0** | **True** | **avg** | **0.001** | **adam** | **0%** | **32** | **0.2** | **47.67%** | **6.3143** | **40** | **48.49** |
| **5** | 1.0 | False | max | 0.0005 | adam | 0% | 32 | 0.2 | 21.67% | 2,216,675.75 | 40 | 61.46 |
| **6** | 1.0 | False | avg | 0.0001 | adam | 20% | 32 | 0.3 | 11.67% | 37,567.67 | 40 | 71.87 |
| **7** | 1.0 | False | avg | 0.0003 | adamw | 0% | 16 | 0.2 | 19.33% | 975,839.81 | 38 | 66.86 |
| **8** | 0.75 | False | avg | 0.0001 | adam | 20% | 16 | 0.3 | 13.00% | 11.75 | 40 | 71.54 |
| **9** | 1.0 | False | avg | 0.0002 | adamw | 20% | 16 | 0.3 | 11.00% | 1,364.33 | 40 | 77.96 |
| **10** | **1.0** | **True** | **avg** | **0.0001** | **adam** | **20%** | **32** | **0.2** | **42.67%** | **1.6739** | **29** | **52.24** |

---

## 3. Konfigurasi Terbaik (Pemenang Validation)

Sesuai aturan di Bagian 5 README utama kelompok, pemilihan model terbaik didasarkan murni pada skor **validation** tertinggi:

* **Trial Terbaik:** **Trial 4**
* **Validation Accuracy:** **47.67%**
* **Validation Loss:** **6.3143**
* **Hyperparameter:**
  * Width Multiplier (`alpha`): `1.0`
  * Varian Backbone: `minimalistic=True` (ReLU standar, tanpa modul SE attention)
  * Pooling Layer: `GlobalAveragePooling2D` (`pooling='avg'`)
  * Learning Rate: `0.001` (Optimizer Adam)
  * Unfreeze Persentase: `0%` (Feature extraction dengan backbone beku)
  * Batch Size: `32`
  * Dropout: `0.2`
* **Kurva Gradient Descent:** Tersimpan di berkas [`history_mobilenetv3large.png`](history_mobilenetv3large.png).

---

## 4. Hasil Evaluasi Test Set (200 Citra)

Evaluasi test set dijalankan secara independen **satu kali** hanya pada konfigurasi terbaik (Trial 4) di akhir notebook:

* **Akurasi Test:** **49.00%** (98 dari 200 citra uji diklasifikasikan dengan benar)
* **Macro F1-Score:** **0.4868**
* **Macro Precision:** **0.5558**
* **Macro Recall:** **0.4900**
* **Jumlah Parameter Model Terbaik:** **1,447,218** (ringkas karena varian `minimalistic=True` tanpa bobot SE dense layer)
* **Waktu Inferensi:** **1.21 ms/gambar** (diuji pada GPU Tesla T4)
* **Confusion Matrix:** Tersimpan di berkas [`confusion_matrix_mobilenetv3large.png`](confusion_matrix_mobilenetv3large.png).

### Laporan Performa per Kategori:

| Kategori | Precision | Recall | F1-Score | Keterangan Performa |
| :--- | :---: | :---: | :---: | :--- |
| **Dinosaur** | **0.9091** | **1.0000** | **0.9524** | **Sangat Unggul** (100% recall, 20/20 benar terdeteksi) |
| **Elephant** | **0.8571** | **0.6000** | **0.7059** | **Sangat Unggul** (presisi tinggi) |
| **Flower** | **0.9091** | **0.5000** | **0.6452** | **Unggul** (presisi tinggi) |
| **Beach** | **0.4545** | **0.7500** | **0.5660** | **Cukup Baik** (recall tinggi) |
| **Bus** | **0.5625** | **0.4500** | **0.5000** | **Cukup** |
| **Horse** | **0.6000** | **0.3000** | **0.4000** | **Sedang** |
| **Food** | **0.2456** | **0.7000** | **0.3636** | **Kurang Unggul** (banyak false positive) |
| **People** | **0.3529** | **0.3000** | **0.3243** | **Kurang Unggul** |
| **Mountain** | **0.5000** | **0.2000** | **0.2857** | **Buruk** (recall sangat rendah, hanya 4/20 benar) |
| **Building** | **0.1667** | **0.1000** | **0.1250** | **Paling Buruk** (hanya 2/20 benar) |

---

## 5. Temuan & Analisis Ilmiah

### A. Mengapa Varian Minimalistic (`minimalistic=True`) Menang Telak?
* Pada dataset kecil berukuran 500 sampel train, aktivasi non-linear Sigmoid pada Squeeze-and-Excitation (SE) attention serta Hard-Swish mengalami saturasi gradien ekstrem jika dilatih tanpa warm-up lr yang bertahap. Hal ini menyebabkan fenomena *exploding validation loss* pada Trial 1, 2, 5, dan 7 ($>970.000$).
* Sebaliknya, varian `minimalistic=True` yang mengganti aktivasi ke fungsi ReLU standar dan menghilangkan modul SE menghasilkan aliran gradien yang stabil, konvergen mulus tanpa ledakan loss, dan mencatatkan akurasi validation tertinggi (47.67% pada Trial 4 dan 42.67% pada Trial 10).

### B. Kegagalan Fine-Tuning pada Varian Standar (*Catastrophic Forgetting*)
* Pada Trial 6, 8, dan 9 ketika 20% layer konvolusi teratas dibuka (*fine-tuning*), akurasi anjlok drastis ke kisaran $\approx 11-13\%$ (setara dengan tebakan acak 10%).
* Hal ini disebabkan oleh perubahan bobot representasi ImageNet yang terlalu agresif akibat sinyal gradien dari dataset kecil yang belum stabil, sehingga model kehilangan kemampuan ekstraksi fitur dasar.

### C. Analisis Kategori Lemah
1. **Building (F1: 0.1250):** Hanya 2 dari 20 citra yang benar. Struktur garis gedung perkotaan berbaur dengan latar belakang langit dan pasir pantai (6 salah diprediksi sebagai Beach).
2. **Mountain (F1: 0.2857):** Hanya 4 dari 20 citra yang benar (Recall 20%). Tekstur lereng gunung bebatuan kelabu dan salju sering disalahartikan sebagai gedung (Building) atau piring makanan (Food).
3. **Food (Precision 24.56%):** Model mengalami bias prediksi yang kuat ke arah kelas Food (ditebak 57 kali dari 200 data). Warna-warni kontras tinggi pada makanan menyerap citra manusia (12), bus (9), dan pemandangan.

---

## 6. Struktur Berkas Folder `model-5_mobilenetv3large/`

```
model-5_mobilenetv3large/
├── README.md                              <- Dokumen ringkasan arsitektur, hasil, dan temuan
├── model-5_mobilenetv3large.ipynb         <- Notebook eksperimen (10 trial + isolasi test set)
├── trials.csv                             <- Rekap log metrik riil 10 trial
├── history_mobilenetv3large.png           <- Kurva loss dan akurasi (gradient descent) model terbaik
├── confusion_matrix_mobilenetv3large.png  <- Matriks kebingungan pada evaluasi 200 citra test
├── catatan_percobaan.md                   <- Catatan komprehensif rencana & analisis trial
└── deskripsi_model.md                     <- Landasan teori arsitektur MobileNetV3Large
```

