# Model-1: ResNet50V2

Dokumentasi resmi untuk **Model-1 (`ResNet50V2`)** pada eksperimen perbandingan 5 arsitektur CNN untuk klasifikasi 10 kategori gambar pada dataset SIMPLIcity (1.000 citra), Tugas Kelompok Neuro Computing (Minggu 11), PENS.

- **Anggota Pasangan:**
  * **Juna**: Bertanggung jawab mengeksekusi **Trial 1 s/d 5**
  * **Fatih**: Bertanggung jawab mengeksekusi **Trial 6 s/d 10**

---

## 1. Arsitektur & Ide Kunci Desain

- **Fungsi Keras:** `keras.applications.ResNet50V2`
- **Bobot Awal:** Pretrained ImageNet (`weights="imagenet"`, `include_top=False`, `pooling="avg"`)
- **Input Resolusi:** $224 \times 224 \times 3$ (Letterbox padding dengan warna rata-rata ImageNet `(124, 116, 104)`)
- **Normalisasi:** `keras.applications.resnet_v2.preprocess_input` (piksel diskalakan ke rentang `[-1, 1]`).
- **Jumlah Parameter Backbone:** $\approx 23.5 \text{ juta}$ parameter (tanpa head klasifikasi).
- **Ide Kunci Arsitektur (He et al., ECCV 2016):**
  1. **Residual Connection (Skip Connection):** Memperkenalkan koneksi *shortcut* yang melewati satu atau beberapa layer. Daripada mempelajari pemetaan fitur secara utuh, layer-layer ini mempelajari fungsi *residual* (selisih). Hal ini mencegah hilangnya gradien (*vanishing gradient*) pada jaringan yang sangat dalam.
  2. **V2 Pre-activation:** Berbeda dengan ResNet versi pertama (V1), versi V2 memindahkan operasi Batch Normalization dan aktivasi ReLU *sebelum* layer konvolusi (pre-activation). Hal ini membuat jalur *shortcut* benar-benar bersih dan gradien mengalir lebih bebas dari loss ke layer awal.
  3. **Bottleneck Architecture:** Menggunakan blok *bottleneck* (konvolusi 1×1, 3×3, 1×1) berturut-turut untuk mengurangi dan kemudian mengembalikan dimensi (channel). Ini membuat komputasi pada 50 layer menjadi jauh lebih efisien.

### Alasan Pemilihan & Kesesuaian dengan Dataset SIMPLIcity

1. **Standar Emas Ekstraksi Fitur:** ResNet50V2 adalah arsitektur yang sangat tangguh dengan kapasitas representasi besar ($\approx 23.5$ juta parameter) yang sudah terbukti di berbagai tugas *Computer Vision*.
2. **Transfer Learning Bebas Hambatan:** Adanya *residual connection* memastikan bahwa saat kita membuka sebagian layer untuk *fine-tuning* di dataset SIMPLIcity, gradien tidak menyusut, sehingga konvergensi lebih cepat dan terarah walau data latih hanya 500 citra.
3. **Pre-activation V2:** Skema V2 meningkatkan keakuratan dan mengurangi *overfitting* yang mana sangat krusial ketika menghadapi dataset bervolume kecil.

---

## 2. Ruang Pencarian & Tabel 10 Trial Hyperparameter (`trials.csv`)

Seluruh 10 konfigurasi hyperparameter di-generate secara deterministik sehingga urutan parameter di akun Juna dan Fatih **identik 100%**:
- **Sidik Konfigurasi:** `b3e0629783c5`
- **Early Stopping:** `monitor="val_loss"`, `patience=8`, `restore_best_weights=True`

| Trial | Buka Layer | Learning Rate | Optimizer | Weight Decay | Dropout | Unit Head | Batch Size | Penanggung Jawab |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 60 layer | $1.00 \times 10^{-3}$ | Adam | 0.0 | 0.3 | 128 | 16 | **Juna** |
| **2** | beku (0 layer) | $1.00 \times 10^{-3}$ | Adam | 0.0 | 0.2 | 128 | 16 | **Juna** |
| **3** | 60 layer | $5.00 \times 10^{-4}$ | Adam | 0.0 | 0.2 | 256 | 16 | **Juna** |
| **4** | 30 layer | $5.00 \times 10^{-4}$ | Adam | 0.0 | 0.2 | 128 | 16 | **Juna** |
| **5** | 30 layer | $5.00 \times 10^{-4}$ | Adam | 0.0 | 0.2 | 0 (direct) | 32 | **Juna** |
| **6** | 60 layer | $1.00 \times 10^{-3}$ | Adam | 0.0 | 0.3 | 256 | 32 | **Fatih** |
| **7** | 30 layer | $1.00 \times 10^{-3}$ | Adam | 0.0 | 0.3 | 128 | 16 | **Fatih** |
| **8** | beku (0 layer) | $1.00 \times 10^{-4}$ | Adam | 0.0 | 0.3 | 256 | 32 | **Fatih** |
| **9** | 60 layer | $1.00 \times 10^{-3}$ | Adam | 0.0 | 0.3 | 0 (direct) | 16 | **Fatih** |
| **10** | beku (0 layer) | $1.00 \times 10^{-4}$ | Adam | 0.0 | 0.5 | 128 | 32 | **Fatih** |

> **Catatan Strategi Layer:**
> * `beku (0 layer)`: Seluruh backbone ImageNet dibekukan, hanya melatih head klasifikasi baru.
> * `30 layer`: Membuka 30 layer teratas dari backbone ResNet50V2.
> * `60 layer`: Membuka 60 layer teratas dari backbone ResNet50V2.
> * **BatchNormalization selalu beku** di semua strategi untuk menjaga kestabilan statistik ImageNet pada ukuran *batch* kecil.

---

## 3. Hasil 10 Trial dan Konfigurasi Terbaik

Gabungan `trials.csv` (Juna trial 1–5, Fatih trial 6–10). Dijalankan di Kaggle Tesla T4. 

| Trial | Buka Layer | Val Acc | Val Loss | Best Epoch | Durasi (s) | Dijalankan |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 60 layer | 0.9433 | 0.1544 | 3 | 330.6 | Juna |
| 2 | beku | 0.9567 | 0.1422 | 4 | 181.9 | Juna |
| 3 | 60 layer | 0.9467 | 0.1589 | 4 | 374.0 | Juna |
| 4 | 30 layer | 0.9433 | 0.1480 | 4 | 395.9 | Juna |
| 5 | 30 layer | 0.9400 | 0.1584 | 20 | 368.1 | Juna |
| 6 | 60 layer | 0.9567 | 0.1849 | 6 | 342.2 | Fatih |
| 7 | 30 layer | 0.9433 | 0.1544 | 3 | 469.5 | Fatih |
| 8 | beku | 0.9533 | 0.1481 | 28 | 360.1 | Fatih |
| **9** | **60 layer** | **0.9633** | **0.1271** | **22** | **513.7** | **Fatih** |
| 10 | beku | 0.9467 | 0.1560 | 39 | 397.5 | Fatih |

**Konfigurasi terbaik: Trial 9**, dipilih dari skor validation (*val_acc*). Trial 9 memiliki skor akurasi tertinggi (0.9633) sekaligus *loss* terendah (0.1271).

| Parameter | Nilai |
| :--- | :--- |
| `buka_layer` | `60` (Fine-tune pada 60 layer teratas backbone) |
| `learning_rate` | $0.001$ |
| `optimizer` | Adam |
| `dropout` | 0.3 |
| `unit_head` | 0 (Tanpa tambahan Dense layer, langsung Dropout) |
| `batch_size` | 16 |
| Epoch terbaik | 22 |

**Temuan dari tuning:**
- **Keseimbangan Fine-Tune:** Strategi membekukan model secara penuh (*freeze*) seperti pada Trial 2 & 8 memiliki performa yang bagus (0.95), namun performa tertinggi justru dicapai ketika 60 layer teratas dibuka (*fine-tune* parsial) dengan arsitektur head yang disederhanakan.
- **Kesederhanaan Head Terbaik:** Trial pemenang (Trial 9) tidak menggunakan penambahan unit dense layer pada kepalanya (`unit_head = 0`), melainkan hanya mengandalkan fitur turunan langsung dari arsitektur ResNet. Ini meminimalkan *overfitting* yang sering terjadi karena jumlah parameter ekstra terlalu besar.
- **Kestabilan Model:** Model ResNet50V2 sangat stabil di beragam nilai hyperparameter, seluruh 10 percobaan konsisten mencapai akurasi validasi > 94%.

Karena Trial 9 dimenangkan oleh Fatih, maka **evaluasi test wajib dijalankan satu kali oleh Fatih**.

---

## 4. Hasil Evaluasi Test (sekali, model trial 9)

> **[ ⚠️ PERHATIAN: Bagian ini akan diisi SETELAH Fatih menjalankan `JALANKAN_TEST = True` di sel terakhir notebook Kaggle / lokalnya ]**

Dijalankan di GPU Kaggle (Tesla T4) dengan model `resnet50v2_trial9.keras` dari output run tuning, tanpa melatih ulang.

| Metrik | Nilai |
| :--- | :--- |
| Akurasi test | [MENUNGGU HASIL TEST] |
| Macro F1 | [MENUNGGU HASIL TEST] |
| Jumlah parameter total | [MENUNGGU HASIL TEST] |
| Parameter dilatih | [MENUNGGU HASIL TEST] |
| Ukuran berkas model | [MENUNGGU HASIL TEST] |
| Waktu inferensi | [MENUNGGU HASIL TEST] |
| Waktu training trial 9 | 513.7 detik |

Grafik matriks kebingungan tersimpan di berkas `confusion_resnet50v2.png` dan ringkasan metrik tersimpan di `hasil_test.json`.

---

## 5. Struktur Berkas di `model-1/`

```text
model-1/
├── model-1_resnet50v2_fatih.ipynb # Notebook utama (atau milik fatih)
├── trials_juna.csv            # Log hasil Trial 1-5 (Juna)
├── trials_fatih.csv           # Log hasil Trial 6-10 (Fatih)
├── fatih_history.json         # Riwayat per epoch Fatih (dan file JSON Juna)
├── trials.csv                 # Tabel gabungan 10 trial (dihasilkan oleh sel gabung)
├── history_resnet50v2.png     # Grafik Loss & Akurasi Gradient Descent model terbaik
├── confusion_resnet50v2.png   # [SEGERA HADIR] Confusion Matrix pada test set
├── hasil_test.json            # [SEGERA HADIR] Ringkasan metrik akhir evaluasi test set
└── README.md                  # Dokumentasi ini
```