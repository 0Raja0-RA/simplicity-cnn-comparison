# Model-4: ConvNeXtTiny

Dokumentasi resmi untuk **Model-4 (`ConvNeXtTiny`)** pada eksperimen perbandingan 5 arsitektur CNN untuk klasifikasi 10 kategori gambar pada dataset SIMPLIcity (1.000 citra), Tugas Kelompok Neuro Computing (Minggu 11), PENS.

* **Anggota Pasangan:**
  * **Ilham**: Bertanggung jawab mengeksekusi **Trial 1 s/d 5**
  * **Arta (Udi Adhahaska Arta Ananta)**: Bertanggung jawab mengeksekusi **Trial 6 s/d 10**

---

## 1. Arsitektur & Ide Kunci Desain

* **Fungsi Keras:** `tf.keras.applications.ConvNeXtTiny`
* **Bobot Awal:** Pretrained ImageNet (`weights="imagenet"`, `include_top=False`, `pooling="avg"`)
* **Input Resolusi:** $224 \times 224 \times 3$ (Letterbox padding dengan warna rata-rata ImageNet `(124, 116, 104)`)
* **Normalisasi:** Diolah secara **internal** oleh layer bawaan `convnext_tiny_prestem_normalization` (memetakan piksel float32 $0-255$ ke mean & std ImageNet). Sesuai aturan nomor 5 panduan kelompok, tidak ditambahkan normalisasi eksternal di luar model.
* **Jumlah Parameter:** $\approx 28.6 \text{ juta}$ parameter backbone ($\approx 27.8 \text{ juta}$ non-head).
* **Ide Kunci Arsitektur (Liu et al., CVPR 2022):**
  1. **Filosofi Modern Pure CNN (ViT-Inspired Design):** ConvNeXt mengadopsi keunggulan struktural Vision Transformer (ViT / Swin) ke dalam arsitektur konvolusi murni tanpa *self-attention*, menghasilkan representasi fitur yang sangat kuat namun tetap mempertahankan efisiensi konvolusi spasial.
  2. **Depthwise Convolution $7 \times 7$:** Menggantikan kernel konvensional $3 \times 3$ dengan kernel besar $7 \times 7$ *depthwise* untuk memperluas receptive field lokal secara signifikan, menyamai mekanisme *local window attention* pada Swin Transformer.
  3. **Inverted Bottleneck Design:** Kanal spasial diperluas 4 kali lipat ($1 \times 4 \times 1$) di dalam block sebelum diproyeksikan kembali, memaksimalkan kapasitas ekstraksi representasi fitur laten.
  4. **Layer Normalization (LN) Menggantikan Batch Normalization (BN):** Mengeliminasi ketergantungan model pada statistik mini-batch yang kerap tidak stabil pada ukuran batch kecil ($16$ atau $32$) saat transfer learning.
  5. **Fewer Activations & Normalizations:** Menggunakan aktivasi **GELU** (*Gaussian Error Linear Unit*) dan memangkas jumlah layer aktivasi menjadi satu per block, mencegah degradasi informasi fitur pada layer-layer dalam.

---

## 2. Ruang Pencarian & Tabel 10 Trial Hyperparameter (`trials.csv`)

Seluruh 10 konfigurasi hyperparameter di-generate secara deterministik menggunakan **`SEED = 42`** di awal notebook sehingga urutan parameter di akun Ilham dan Arta **identik 100%**:
* **Sidik Konfigurasi:** `e93feaa4cb03`
* **Batas Maksimal Epoch:** 40 epoch
* **Early Stopping:** `monitor="val_loss"`, `patience=8`, `restore_best_weights=True`

| Trial | Buka Layer | Learning Rate | Optimizer | Weight Decay | Dropout | Unit Head | Batch Size | Penanggung Jawab |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | beku | $4.45 \times 10^{-4}$ | AdamW | $1 \times 10^{-4}$ | 0.41 | 256 | 16 | **Ilham** |
| **2** | beku | $2.76 \times 10^{-3}$ | AdamW | $1 \times 10^{-3}$ | 0.44 | 256 | 16 | **Ilham** |
| **3** | stage_2+3 | $3.00 \times 10^{-5}$ | Adam | 0.0 | 0.48 | 256 | 32 | **Ilham** |
| **4** | stage_3 | $3.80 \times 10^{-5}$ | AdamW | $1 \times 10^{-4}$ | 0.37 | 0 (direct) | 32 | **Ilham** |
| **5** | beku | $1.67 \times 10^{-3}$ | Adam | 0.0 | 0.43 | 256 | 32 | **Ilham** |
| **6** | stage_3 | $1.83 \times 10^{-4}$ | Adam | 0.0 | 0.43 | 256 | 32 | **Arta** |
| **7** | beku | $4.89 \times 10^{-4}$ | Adam | 0.0 | 0.25 | 0 (direct) | 32 | **Arta** |
| **8** | stage_2+3 | $9.30 \times 10^{-5}$ | Adam | 0.0 | 0.30 | 256 | 32 | **Arta** |
| **9** | stage_3 | $4.10 \times 10^{-5}$ | AdamW | $1 \times 10^{-4}$ | 0.24 | 256 | 16 | **Arta** |
| **10** | beku | $9.76 \times 10^{-4}$ | Adam | 0.0 | 0.33 | 0 (direct) | 32 | **Arta** |

> **Catatan Strategi Layer:**
> * `beku`: Seluruh backbone ImageNet dibekukan, hanya melatih head klasifikasi baru.
> * `stage_3`: Membuka stage konvolusi teratas (`convnext_tiny_stage_3_*`) dan layer normalisasi akhir (`layer_normalization`).
> * `stage_2+3`: Membuka stage 2 dan 3 untuk penyesuaian representasi semantik tingkat tinggi.

---

## 3. Alur Kerja Eksekusi di Kaggle

### A. Untuk Arta (Trial 6 s/d 10)
1. Unggah berkas `model-4_convnext_tiny.ipynb` ke Kaggle.
2. Di panel kanan Kaggle (*Notebook Settings*):
   * **Accelerator**: GPU T4 x2 atau P100.
   * **Internet**: **ON**.
   * Klik **Add Input**, cari dan tambahkan dataset: **`yonobengkel12/simplicity-224-letterbox`**.
3. Pastikan di Sel 8:
   ```python
   NAMA = "arta"
   TRIAL_SAYA = [6, 7, 8, 9, 10]
   ```
4. Pilih **Save Version > Save & Run All** (atau jalankan sel secara berurutan).
5. Setelah selesai, unduh berkas `trials_arta.csv` dan `history_arta.json` dari tab **Output Kaggle**.
6. Simpan berkas hasil unduhan ke folder `model-4_convnext_tiny/` di laptop.

### B. Untuk Ilham (Trial 1 s/d 5)
1. Buka notebook yang sama di akun Kaggle Ilham.
2. Atur di Sel 8:
   ```python
   NAMA = "ilham"
   TRIAL_SAYA = [1, 2, 3, 4, 5]
   ```
3. Jalankan hingga selesai dan unduh `trials_ilham.csv` serta `history_ilham.json`.

### C. Penggabungan & Evaluasi Test Set Akhir
1. Letakkan `trials_ilham.csv` dan `trials_arta.csv` di folder `model-4_convnext_tiny/`.
2. Jalankan Sel 12 (*Gabung Hasil Pasangan*):
   * Otomatis membuat `trials.csv` gabungan lengkap 10 trial.
   * Otomatis menyimpan grafik kurva konvergensi gradient descent `history_convnext_tiny.png`.
   * Menampilkan model pemenang (*val_acc tertinggi / val_loss terendah*).
3. Pemilik trial terbaik mengaktifkan `JALANKAN_TEST = True` di Sel 14 untuk evaluasi akhir pada 200 citra uji (Akurasi, Macro F1, Confusion Matrix, dan Latensi Inferensi).

---

## 4. Struktur Berkas di `model-4_convnext_tiny/`

```text
model-4_convnext_tiny/
├── model-4_convnext_tiny.ipynb  # Notebook utama untuk Kaggle
├── trials_ilham.csv             # Log hasil Trial 1-5 (Ilham)
├── trials_arta.csv              # Log hasil Trial 6-10 (Arta)
├── history_ilham.json           # Riwayat per epoch Trial 1-5
├── history_arta.json            # Riwayat per epoch Trial 6-10
├── trials.csv                   # Tabel gabungan 10 trial (dihasilkan oleh sel gabung)
├── history_convnext_tiny.png    # Grafik Loss & Akurasi Gradient Descent model terbaik
├── confusion_convnext_tiny.png  # Confusion Matrix pada test set
├── hasil_test.json              # Ringkasan metrik akhir evaluasi test set
└── README.md                    # Dokumentasi ini
```

