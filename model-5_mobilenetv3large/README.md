# Model-5: MobileNetV3Large

Dokumentasi resmi untuk **Model-5 (`MobileNetV3Large`)** pada eksperimen perbandingan 5 arsitektur CNN untuk klasifikasi 10 kategori gambar pada dataset SIMPLIcity (1.000 citra), Tugas Kelompok Neuro Computing (Minggu 11), PENS.

---

## 1. Arsitektur & Ide Kunci Desain

* **Fungsi Keras:** `keras.applications.MobileNetV3Large`
* **Bobot Awal:** Pretrained ImageNet (`weights="imagenet"`)
* **Input Resolusi:** $224 \times 224 \times 3$ (Letterbox padding dengan warna rata-rata ImageNet `(124, 116, 104)`)
* **Normalisasi Input:** Skala piksel $[-1, 1]$ ditangani langsung di dalam model via layer internal `Rescaling(scale=1/127.5, offset=-1.0)` dengan mengaktifkan `include_preprocessing=True`.
* **Jumlah Parameter:** $3.005.962$ parameter (pada $\alpha = 1.0$; $1.447.218$ jika varian `minimalistic=True`)
* **Ide Kunci Arsitektur:**
  1. **Depthwise Separable Convolutions:** Memisahkan penyaringan spasial ($3 \times 3$ / $5 \times 5$ Depthwise) dan kombinasi representasi kanal ($1 \times 1$ Pointwise), memangkas beban komputasi FLOPs dan parameter hingga $\approx 85-90\%$ dibanding konvolusi standar.
  2. **Squeeze-and-Excitation (SE) Attention Block:** Modul kanal adaptif yang menimbang tingkat kepentingan fitur per-kanal melalui mekanisme Squeeze (*Global Average Pooling*) dan Excitation (*2 Dense Layers* dengan aktivasi non-linear ReLU & Sigmoid).
  3. **Hard-Swish ($h\text{-swish}$):** Fungsi aktivasi non-linear efisien $h\text{-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$ yang mengaproksimasi Swish tanpa komputasi eksponensial mahal.

### Alasan Pemilihan & Kesesuaian dengan Dataset SIMPLIcity
1. **Skala Parameter Ideal untuk Dataset Ringkas:** Dataset SIMPLIcity hanya memiliki 500 citra training (50 citra per kelas). Model raksasa seperti ResNet152 atau DenseNet201 rentan mengalami *overfitting*. MobileNetV3Large yang ringkas ($\approx 3.0\text{M}$ parameter) menawarkan regularisasi kapasitas yang alami (*strong inductive bias*).
2. **Efisiensi & Latensi Ultra Rendah:** Dirancang untuk inferensi cepat pada perangkat komputasi terbatas dengan latensi rata-rata hanya **1.47 ms per citra** di GPU Tesla T4.

---

## 2. Log 10 Trial Hyperparameter Tuning (`trials.csv`)

Seluruh 10 trial dijalankan pada GPU Kaggle (Tesla T4) dengan protokol fixed seed (`SEED = 42`), early stopping `patience=8`, dan batas maksimum 40 epoch:

| Trial | Alpha | Minimalistic | Pooling | Learning Rate | Optimizer | Unfreeze % | Batch Size | Dropout | Val Acc (%) | Val Loss | Best Epoch | Train Time (s) | Keterangan |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **1.0** | **False** | **avg** | **0.001** | **adam** | **0%** | **32** | **0.2** | **97.00%** | **0.1187** | **29** | **57.69** | **PEMENANG (Val Acc Tertinggi)** |
| **2** | 1.0 | False | avg | 0.0005 | adam | 0% | 32 | 0.3 | 96.00% | 0.1287 | 39 | 57.69 | Konvergen sangat stabil |
| **3** | 0.75 | False | avg | 0.001 | adam | 0% | 32 | 0.2 | 95.67% | 0.1314 | 40 | 53.71 | Model lebih ramping (alpha 0.75) |
| **4** | 1.0 | True | avg | 0.001 | adam | 0% | 32 | 0.2 | 95.67% | 0.1248 | 40 | 48.22 | Minimalistic (tanpa SE & H-Swish) |
| **5** | 1.0 | False | max | 0.0005 | adam | 0% | 32 | 0.2 | 92.33% | 0.7639 | 35 | 57.94 | Global Max Pooling |
| **6** | 1.0 | False | avg | 0.0001 | adam | 20% | 32 | 0.3 | 96.33% | 0.1135 | 11 | 36.53 | Fine-tuning 20% (Loss Terendah) |
| **7** | 1.0 | False | avg | 0.0003 | adamw | 0% | 16 | 0.2 | 95.33% | 0.1367 | 39 | 63.27 | AdamW optimizer, batch 16 |
| **8** | 0.75 | False | avg | 0.0001 | adam | 20% | 16 | 0.3 | 96.67% | 0.1219 | 17 | 43.80 | Runner-up (0.75 + fine-tuning) |
| **9** | 1.0 | False | avg | 0.0002 | adamw | 20% | 16 | 0.3 | 95.67% | 0.1166 | 4 | 27.87 | Early convergence epoch 4 |
| **10** | 1.0 | True | avg | 0.0001 | adam | 20% | 32 | 0.2 | 95.00% | 0.1350 | 16 | 33.84 | Minimalistic + fine-tuning |

---

## 3. Konfigurasi Terbaik (Pemenang Validation)

Sesuai aturan di Bagian 5 README utama kelompok, pemilihan model terbaik didasarkan murni pada skor **validation** tertinggi:

* **Trial Terbaik:** **Trial 1**
* **Validation Accuracy:** **97.00%**
* **Validation Loss:** **0.1187**
* **Epoch Terbaik:** 29 (dari 40 epoch, early stopping berhenti di epoch 37)
* **Hyperparameter:**
  * Width Multiplier (`alpha`): `1.0`
  * Varian Backbone: `minimalistic=False` (arsitektur standar MobileNetV3Large lengkap dengan modul SE attention dan Hard-Swish)
  * Pooling Layer: `GlobalAveragePooling2D` (`pooling='avg'`)
  * Learning Rate: `0.001` (Optimizer Adam)
  * Unfreeze Persentase: `0%` (Feature extraction dengan backbone beku 100%)
  * Batch Size: `32`
  * Dropout: `0.2`
* **Kurva Gradient Descent:** Tersimpan di berkas [`history_mobilenetv3large.png`](history_mobilenetv3large.png) dan riwayat nilai numerik tersimpan di [`history_oka.json`](history_oka.json).

---

## 4. Hasil Evaluasi Test Set (200 Citra)

Evaluasi test set dijalankan secara independen **satu kali** hanya pada konfigurasi terbaik (Trial 1) di sel terakhir notebook:

* **Akurasi Test:** **98.00%** (196 dari 200 citra uji diklasifikasikan dengan benar; hanya 4 citra salah klasifikasi)
* **Macro F1-Score:** **0.9802**
* **Macro Precision:** **0.9830**
* **Macro Recall:** **0.9800**
* **Jumlah Parameter Total:** **3.005.962**
* **Parameter Dilatih:** **12.810** (hanya layer Dense output $1280 \times 10 + 10$)
* **Ukuran Berkas Model:** **11.5 MB**
* **Waktu Inferensi:** **1.47 ms/gambar** (diuji pada GPU Tesla T4)
* **Confusion Matrix:** Tersimpan di berkas [`confusion_matrix_mobilenetv3large.png`](confusion_matrix_mobilenetv3large.png).

### Laporan Performa per Kategori:

| Kategori | Precision | Recall | F1-Score | Keterangan Performa |
| :--- | :---: | :---: | :---: | :--- |
| **Beach** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** (20/20 benar terdeteksi, tanpa false positive) |
| **Bus** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** (20/20 benar terdeteksi) |
| **Dinosaur** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** (20/20 benar terdeteksi) |
| **Flower** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** (20/20 benar terdeteksi) |
| **Food** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** (20/20 benar terdeteksi) |
| **Horse** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** (20/20 benar terdeteksi) |
| **Mountain** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** (20/20 benar terdeteksi) |
| **Elephant** | **1.0000** | **0.9500** | **0.9744** | **Sangat Unggul** (19/20 benar, 1 salah diprediksi People) |
| **Building** | **1.0000** | **0.8500** | **0.9189** | **Sangat Baik** (17/20 benar, 3 salah diprediksi People) |
| **People** | **0.8333** | **1.0000** | **0.9091** | **Sangat Baik** (20/20 benar terdeteksi, menyerap 4 citra luar) |

---

## 5. Temuan & Analisis Ilmiah

### A. Keberhasilan Normalisasi & Kekuatan Modul Squeeze-and-Excitation (SE)
* Pada iterasi awal proyek, model mengalami kegagalan konvergensi akibat Keras 3 `include_preprocessing=False` yang mematikan layer `Rescaling(scale=1/127.5, offset=-1)` internal, sementara `mobilenet_v3.preprocess_input` di Keras 3 adalah *no-op*. Hal tersebut menyebabkan piksel mentah $[0, 255]$ masuk ke bobot ImageNet.
* Setelah perbaikan dengan `include_preprocessing=True`, rentang input dinormalisasi sempurna ke $[-1, 1]$. Hasilnya, **seluruh 10 trial konvergen di atas 92% - 97%**.
* Modul **Squeeze-and-Excitation (SE) Attention** membuktikan keunggulannya: mekanisme pembobotan kanal dinamis memberikan fitur diskriminatif yang sangat tajam sehingga **7 dari 10 kelas mencapai presisi dan recall sempurna 100% (F1 = 1.0000)**.

### B. Feature Extraction Murni vs Fine-Tuning
* **Feature Extraction (Trial 1):** Mencapai validation accuracy tertinggi ($97.00\%$) dan test accuracy $98.00\%$. Pembekuan backbone konvolusi ($100\%$ beku) melindungi bobot representasi kaya dari ImageNet terhadap distorsi dari dataset kecil SIMPLIcity (500 sampel train).
* **Fine-Tuning (Trial 6, 8, 9):** Berbeda dengan run unnormalized sebelumnya yang mengalami *catastrophic forgetting*, fine-tuning dengan learning rate kecil ($1\times 10^{-4}$ s.d. $2\times 10^{-4}$) berjalan sangat kompetitif. Trial 8 mencapai val accuracy $96.67\%$ dan Trial 6 mencatatkan val loss terendah di seluruh eksperimen ($0.1135$). Hal ini membuktikan kestabilan representasi MobileNetV3Large.

### C. Analisis Kesalahan Klasifikasi pada Confusion Matrix (4 dari 200 Citra)
Dari 200 citra uji (20 citra per kelas), **hanya 4 citra** yang salah diprediksi, dan seluruh kesalahan bermuara pada satu kelas: **People**:
1. **3 Citra Building $\rightarrow$ Diprediksi People:**  
   Pada dataset SIMPLIcity, foto monumen, candi, dan arsitektur bersejarah kerap kali memuat kerumunan wisatawan di pelataran gedung. Model menangkap fitur figur manusia sebagai sinyal semantik yang menonjol dibanding struktur statis bangunan di belakangnya.
2. **1 Citra Elephant $\rightarrow$ Diprediksi People:**  
   Foto gajah pada dataset SIMPLIcity sering kali berupa gajah atraksi/pekerja yang ditunggangi oleh pawang manusia (*mahout*) di atas punggungnya.
3. **Dampak pada Kelas People:**  
   Recall kelas People bernilai **100%** (20/20 citra asli People berhasil diidentifikasi tanpa ada yang terlewat). Namun, karena menyerap 4 citra dari kelas Building dan Elephant, presisi kelas People menjadi $\frac{20}{20 + 3 + 1} = 83.33\%$.

---

## 6. Struktur Berkas Folder `model-5_mobilenetv3large/`

```
model-5_mobilenetv3large/
├── README.md                              <- Dokumen resmi arsitektur, log 10 trial, hasil test, dan temuan
├── model-5_mobilenetv3large.ipynb         <- Notebook eksperimen (10 trial tuning + sel isolasi test set)
├── trials.csv                             <- Rekap log metrik riil 10 trial hyperparameter tuning
├── history_mobilenetv3large.png           <- Grafik kurva loss dan akurasi (gradient descent) model terbaik
├── history_oka.json                       <- Rekap numerik per-epoch metrik training & validation 10 trial
├── confusion_matrix_mobilenetv3large.png  <- Matriks kebingungan evaluasi 200 citra test set
├── hasil_test.json                        <- Ringkasan metrik akhir evaluasi test set
├── catatan_percobaan.md                   <- Catatan komprehensif rencana, log riil, dan analisis trial
└── deskripsi_model.md                     <- Landasan teori arsitektur MobileNetV3Large
```
