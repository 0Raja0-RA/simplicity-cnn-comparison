# Catatan Percobaan Model-5: MobileNetV3Large

Dokumen ini mencatat rincian desain eksperimen, perbedaan parameter pada setiap run (Trial 1 sampai 10), hasil evaluasi riil, serta analisis mendalam mengenai konfigurasi dan analisis kebingungan kelas pada dataset SIMPLIcity.

---

## 1. Rincian Perbedaan Tiap Percobaan (Trial 1 - 10)

Setiap percobaan menguji variasi parameter dari fungsi Keras `keras.applications.MobileNetV3Large` (`alpha`, `minimalistic`, `pooling`) yang dipadukan dengan strategi pelatihan (`lr`, `optimizer`, `unfreeze_pct`, `batch_size`, `dropout`):

* **Trial 1 (Baseline & Pemenang):** `alpha=1.0, minimalistic=False, pooling='avg', lr=1e-3, opt=adam, unfreeze=0%, bs=32, drop=0.2` (Arsitektur standar lengkap dengan SE attention dan Hard-Swish, backbone beku murni).
* **Trial 2:** `lr=5e-4, drop=0.3` (Mengecilkan learning rate dan menaikkan dropout untuk menguji keteraturan konvergensi).
* **Trial 3:** `alpha=0.75` (Width multiplier lebih kecil, menguji kapasitas model lebih ramping dengan parameter $\approx 2.0\text{M}$).
* **Trial 4:** `minimalistic=True` (Menguji varian tanpa modul SE attention dan menggunakan ReLU standar).
* **Trial 5:** `pooling='max'` (Menguji GlobalMaxPooling vs GlobalAveragePooling pada pemetaan representasi spasial).
* **Trial 6:** `unfreeze=20%, lr=1e-4, drop=0.3` (Fine-tuning 20% layer konvolusi teratas dengan learning rate konservatif).
* **Trial 7:** `opt=adamw, bs=16, lr=3e-4` (Menguji optimizer AdamW dengan decoupled weight decay dan batch size 16).
* **Trial 8:** `alpha=0.75, unfreeze=20%, bs=16, lr=1e-4` (Kombinasi model ramping $\alpha=0.75$ dipadu dengan fine-tuning 20%).
* **Trial 9:** `opt=adamw, unfreeze=20%, bs=16, lr=2e-4` (Fine-tuning 20% layer teratas dengan optimizer AdamW).
* **Trial 10:** `minimalistic=True, unfreeze=20%, lr=1e-4` (Fine-tuning pada varian minimalistic untuk pembanding Trial 6).

---

## 2. Tabel Ringkasan Rencana 10 Trial

| Trial | Alpha | Minimalistic | Pooling | Learning Rate | Optimizer | Unfreeze % | Batch Size | Dropout | Fokus Eksperimen |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `1.0` | `False` | `'avg'` | `1e-3` | `adam` | `0%` | `32` | `0.2` | Baseline feature extraction standar |
| **2** | `1.0` | `False` | `'avg'` | `5e-4` | `adam` | `0%` | `32` | `0.3` | Pengurangan LR & kenaikan dropout |
| **3** | `0.75`| `False` | `'avg'` | `1e-3` | `adam` | `0%` | `32` | `0.2` | Uji width multiplier lebih ramping |
| **4** | `1.0` | `True`  | `'avg'` | `1e-3` | `adam` | `0%` | `32` | `0.2` | Uji ketiadaan modul SE & Hard-Swish |
| **5** | `1.0` | `False` | `'max'` | `5e-4` | `adam` | `0%` | `32` | `0.2` | Uji Global Max Pooling vs Average Pooling |
| **6** | `1.0` | `False` | `'avg'` | `1e-4` | `adam` | `20%`| `32` | `0.3` | Uji fine-tuning 20% layer teratas |
| **7** | `1.0` | `False` | `'avg'` | `3e-4` | `adamw`| `0%` | `16` | `0.2` | Uji optimizer AdamW & batch size 16 |
| **8** | `0.75`| `False` | `'avg'` | `1e-4` | `adam` | `20%`| `16` | `0.3` | Kombinasi alpha 0.75 + fine-tuning 20% |
| **9** | `1.0` | `False` | `'avg'` | `2e-4` | `adamw`| `20%`| `16` | `0.3` | Fine-tuning 20% dipadukan dengan AdamW |
| **10**| `1.0` | `True`  | `'avg'` | `1e-4` | `adam` | `20%`| `32` | `0.2` | Fine-tuning pada varian minimalistic |

---

## 3. Log Hasil Percobaan Riil (`trials.csv`)

Hasil eksekusi 10 trial pada GPU Kaggle (Tesla T4) dengan normalisasi internal `include_preprocessing=True` aktif:

| Trial | Alpha | Minimalistic | Pooling | LR | Optimizer | Unfreeze % | Batch | Dropout | Val Acc (%) | Val Loss | Best Epoch | Train Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **1.0** | **False** | **avg** | **0.001** | **adam** | **0** | **32** | **0.2** | **97.00%** | **0.1187** | **29** | **57.69** |
| **2** | 1.0 | False | avg | 0.0005 | adam | 0 | 32 | 0.3 | 96.00% | 0.1287 | 39 | 57.69 |
| **3** | 0.75 | False | avg | 0.001 | adam | 0 | 32 | 0.2 | 95.67% | 0.1314 | 40 | 53.71 |
| **4** | 1.0 | True | avg | 0.001 | adam | 0 | 32 | 0.2 | 95.67% | 0.1248 | 40 | 48.22 |
| **5** | 1.0 | False | max | 0.0005 | adam | 0 | 32 | 0.2 | 92.33% | 0.7639 | 35 | 57.94 |
| **6** | 1.0 | False | avg | 0.0001 | adam | 20 | 32 | 0.3 | 96.33% | 0.1135 | 11 | 36.53 |
| **7** | 1.0 | False | avg | 0.0003 | adamw | 0 | 16 | 0.2 | 95.33% | 0.1367 | 39 | 63.27 |
| **8** | 0.75 | False | avg | 0.0001 | adam | 20 | 16 | 0.3 | 96.67% | 0.1219 | 17 | 43.80 |
| **9** | 1.0 | False | avg | 0.0002 | adamw | 20 | 16 | 0.3 | 95.67% | 0.1166 | 4 | 27.87 |
| **10**| 1.0 | True | avg | 0.0001 | adam | 20 | 32 | 0.2 | 95.00% | 0.1350 | 16 | 33.84 |

---

## 4. Konfigurasi Terbaik (Pemenang Validation)

* **Trial Pemenang:** **Trial 1**
* **Validation Accuracy:** **97.00%** (`val_acc = 0.9700`)
* **Validation Loss:** **0.1187**
* **Epoch Terbaik:** 29 (dari 40 epoch, early stopping berhenti di epoch 37)
* **Spesifikasi Hyperparameter:**
  * `alpha`: 1.0
  * `minimalistic`: **False** (arsitektur standar MobileNetV3Large dengan modul SE attention dan aktivasi Hard-Swish)
  * `pooling`: 'avg' (GlobalAveragePooling)
  * `lr`: $1 \times 10^{-3}$ (Adam)
  * `unfreeze_pct`: 0% (Feature extraction dengan backbone beku)
  * `batch_size`: 32
  * `dropout`: 0.2
* **Grafik Gradient Descent:** Tersimpan di `history_mobilenetv3large.png` dan seluruh data per-epoch tersimpan di `history_oka.json`.

---

## 5. Evaluasi Akhir Test Set (200 Citra)

Evaluasi dilakukan satu kali pada konfigurasi terbaik (Trial 1) menggunakan data test independen (20 citra per kelas):

* **Akurasi Test:** **98.00%** (196 dari 200 citra diklasifikasikan dengan benar)
* **Macro F1-Score:** **0.9802**
* **Macro Precision:** **0.9830**
* **Macro Recall:** **0.9800**
* **Jumlah Parameter Total:** **3.005.962**
* **Parameter Dilatih:** **12.810**
* **Waktu Inferensi:** **1.47 ms/gambar** (GPU Tesla T4)
* **Confusion Matrix:** Tersimpan di `confusion_matrix_mobilenetv3large.png`

### Laporan Performa per Kategori:

| Kategori | Precision | Recall | F1-Score | Status Performa | Detail Evaluasi (Benar/Total) |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Beach** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** | 20 / 20 |
| **Bus** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** | 20 / 20 |
| **Dinosaur** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** | 20 / 20 |
| **Flower** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** | 20 / 20 |
| **Food** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** | 20 / 20 |
| **Horse** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** | 20 / 20 |
| **Mountain** | **1.0000** | **1.0000** | **1.0000** | **Sempurna** | 20 / 20 |
| **Elephant** | **1.0000** | **0.9500** | **0.9744** | **Sangat Unggul** | 19 / 20 (1 diprediksi People) |
| **Building** | **1.0000** | **0.8500** | **0.9189** | **Sangat Baik** | 17 / 20 (3 diprediksi People) |
| **People** | **0.8333** | **1.0000** | **0.9091** | **Sangat Baik** | 20 / 20 (menyerap 4 citra) |

---

## 6. Analisis Performa & Pembahasan Ilmiah

### A. Analisis Efektivitas Konfigurasi
1. **Keunggulan Feature Extraction (Trial 1 - 97.00% Val Acc, 98.00% Test Acc):**
   * Mengunci bobot backbone konvolusi ImageNet dan hanya melatih 12.810 parameter pada Dense head terbukti menjadi strategi paling optimal untuk SIMPLIcity (500 sampel train).
   * Fitur generalisasi ImageNet sangat kaya, sehingga model terhindar dari *overfitting* yang umum terjadi pada dataset berskala kecil.
2. **Kinerja Fine-Tuning yang Stabil (Trial 6, 8, 9):**
   * Setelah skala input dinormalisasi dengan benar ke $[-1, 1]$, proses *fine-tuning* pada 20% layer teratas berjalan sangat stabil.
   * Trial 8 mencapai akurasi validation 96.67%, dan Trial 6 mencapai validation loss terendah di seluruh eksperimen ($0.1135$ pada epoch 11). Ini membuktikan bahwa *catastrophic forgetting* yang terjadi pada percobaan lama murni akibat unnormalized inputs, bukan kelemahan arsitektur MobileNetV3Large.
3. **Kelemahan Relatif Global Max Pooling (Trial 5 - 92.33% Val Acc, Loss 0.7639):**
   * Trial 5 yang menggunakan `pooling='max'` mencatatkan akurasi terendah di antara 10 trial (92.33%) dengan loss terbesar (0.7639).
   * Hal ini disebabkan Max Pooling hanya mempertahankan satu nilai piksel paling ekstrem di setiap peta fitur, yang rentan terdistorsi oleh noise latar belakang atau variasi pencahayaan, berbeda dengan Average Pooling yang merangkum konteks global citra.

### B. Analisis Kesalahan Klasifikasi (Confusion Matrix)
Dari 200 citra uji independen, model hanya melakukan **4 kesalahan** di seluruh kelas:
1. **3 Citra Building Terklasifikasi sebagai People:**
   * Pada foto monumen atau bangunan bersejarah di dataset SIMPLIcity, seringkali terdapat turis atau peziarah yang berkerumun di pelataran bangunan.
   * Model mendeteksi fitur biologis/manusia yang sangat tajam dan mengatribusikan citra tersebut ke kelas People.
2. **1 Citra Elephant Terklasifikasi sebagai People:**
   * Citra gajah dalam SIMPLIcity umumnya adalah gajah asia yang ditunggangi pawang (*mahout*). Keberadaan manusia di atas gajah memicu salah satu prediksi ke kelas People.
3. **Karakteristik Kelas People:**
   * Recall kelas People bernilai sempurna **100%** (20/20 citra People asli berhasil ditebak benar).
   * Namun karena menerima 4 false positive dari Building dan Elephant, nilai presisi menjadi $\frac{20}{24} = 83.33\%$.

---

## 7. Catatan Teknis Resolusi Bug Normalisasi Keras 3

* **Investigasi Akar Masalah:**
  Pada implementasi awal, notebook mematikan normalisasi internal model (`include_preprocessing=False`) dan memanggil `mobilenet_v3.preprocess_input(x)`. Di Keras 3, fungsi eksternal tersebut adalah *no-op* passthrough, menyebabkan piksel mentah $[0, 255]$ langsung dioperasikan pada bobot ImageNet yang dilatih pada $[-1, 1]$. Hal tersebut memicu exploding loss pada fungsi aktivasi Sigmoid (modul SE) dan Hard-Swish.
* **Tindakan Koreksi:**
  Menyetel `include_preprocessing=True` pada deklarasi `keras.applications.MobileNetV3Large` mengaktifkan layer `Rescaling(scale=1/127.5, offset=-1.0)` bawaan secara otomatis.
* **Hasil Pasca-Koreksi:**
  Akurasi validasi melonjak dari $\approx 18-47\%$ menjadi **$95-97\%$**, dan akurasi test set melesat ke **$98.00\%$ (Macro F1: 0.9802)**, menempatkan MobileNetV3Large sebagai salah satu model terkuat dan paling efisien dalam studi perbandingan kelompok.
