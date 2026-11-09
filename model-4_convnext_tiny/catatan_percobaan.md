# Catatan Percobaan Model-4: ConvNeXtTiny

Dokumen ini mencatat rincian desain eksperimen, perbedaan parameter pada setiap run (Trial 1 sampai 10), hasil evaluasi riil, serta analisis mendalam mengenai konfigurasi dan dinamika konvergensi gradien descent pada dataset SIMPLIcity (1.000 citra).

* **Tim Model-4:**
  * **Ilham**: Bertanggung jawab mengeksekusi **Trial 1 s/d 5**
  * **Arta (Udi Adhahaska Arta Ananta)**: Bertanggung jawab mengeksekusi **Trial 6 s/d 10**

---

## 1. Rincian Perbedaan Tiap Percobaan (Trial 1 - 10)

Seluruh 10 konfigurasi di-generate secara deterministik menggunakan **`SEED = 42`** di awal notebook dengan sidik konfigurasi **`e93feaa4cb03`**:

* **Trial 1 (Ilham):** `buka_layer='beku', lr=4.45e-4, opt=adamw, wd=1e-4, drop=0.41, head=256, bs=16` (Backbone beku, AdamW decoupled weight decay).
* **Trial 2 (Ilham):** `buka_layer='beku', lr=2.76e-3, opt=adamw, wd=1e-3, drop=0.44, head=256, bs=16` (Learning rate agresif pada mode beku).
* **Trial 3 (Ilham):** `buka_layer='stage_2+3', lr=3.00e-5, opt=adam, wd=0.0, drop=0.48, head=256, bs=32` (Fine-tuning mendalam stage 2 dan 3).
* **Trial 4 (Ilham):** `buka_layer='stage_3', lr=3.80e-5, opt=adamw, wd=1e-4, drop=0.37, head=0, bs=32` (Fine-tuning stage 3 dengan direct head).
* **Trial 5 (Ilham):** `buka_layer='beku', lr=1.67e-3, opt=adam, wd=0.0, drop=0.43, head=256, bs=32` (Feature extraction dengan optimizer Adam).
* **Trial 6 (Arta):** `buka_layer='stage_3', lr=1.83e-4, opt=adam, wd=0.0, drop=0.43, head=256, bs=32` (Fine-tuning stage konvolusi teratas + intermediate head).
* **Trial 7 (Arta):** `buka_layer='beku', lr=4.89e-4, opt=adam, wd=0.0, drop=0.25, head=0, bs=32` (Feature extraction langsung dari pooling ke softmax tanpa hidden layer).
* **Trial 8 (Arta):** `buka_layer='stage_2+3', lr=9.31e-5, opt=adam, wd=0.0, drop=0.30, head=256, bs=32` (Fine-tuning adaptasi fitur semantik menengah hingga tinggi).
* **Trial 9 (Arta):** `buka_layer='stage_3', lr=4.08e-5, opt=adamw, wd=1e-5, drop=0.24, head=256, bs=16` (Fine-tuning stage 3 dengan AdamW & batch size 16).
* **Trial 10 (Arta):** `buka_layer='beku', lr=9.76e-4, opt=adam, wd=0.0, drop=0.33, head=0, bs=32` (Mode beku direct head dengan learning rate mendekati 1e-3).

---

## 2. Log Hasil Percobaan Riil Arta (`trials_arta.csv`)

Hasil eksekusi Trial 6 s/d 10 pada GPU Kaggle (Tesla T4) dengan total durasi pelatihan $\approx 34.7$ menit:

| Trial | Buka Layer | Learning Rate | Optimizer | Weight Decay | Dropout | Unit Head | Batch | Val Acc (%) | Val Loss | Best Epoch | Epoch Total | Durasi (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6** | **stage_3** | **$1.83 \times 10^{-4}$** | **adam** | **0.0** | **0.43** | **256** | **32** | **98.67%** | **0.0580** | **15** | **23** | **316.0 s (5.3 mnt)** |
| **7** | beku | $4.89 \times 10^{-4}$ | adam | 0.0 | 0.25 | 0 | 32 | 98.00% | 0.0791 | 40 | 40 | 364.1 s (6.1 mnt) |
| **8** | stage_2+3 | $9.31 \times 10^{-5}$ | adam | 0.0 | 0.30 | 256 | 32 | 98.00% | 0.0653 | 15 | 23 | 504.6 s (8.4 mnt) |
| **9** | stage_3 | $4.08 \times 10^{-5}$ | adamw | $1 \times 10^{-5}$ | 0.24 | 256 | 16 | 98.67% | 0.0627 | 39 | 40 | 534.2 s (8.9 mnt) |
| **10** | beku | $9.76 \times 10^{-4}$ | adam | 0.0 | 0.33 | 0 | 32 | 97.67% | 0.0665 | 40 | 40 | 365.6 s (6.1 mnt) |

Di antara trial 6-10, trial 6 memiliki val_loss terendah (0.0580).

---

## 2b. Log Hasil Percobaan Riil Ilham (`trials_ilham.csv`)

Hasil eksekusi Trial 1 s/d 5 pada GPU Kaggle (Tesla T4) dengan total durasi pelatihan $\approx 32.9$ menit:

| Trial | Buka Layer | Learning Rate | Optimizer | Weight Decay | Dropout | Unit Head | Batch | Val Acc (%) | Val Loss | Best Epoch | Epoch Total | Durasi (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | beku | $4.45 \times 10^{-4}$ | adamw | $1 \times 10^{-4}$ | 0.41 | 256 | 16 | 98.67% | 0.0544 | 15 | 23 | 211.7 s (3.5 mnt) |
| **2** | beku | $2.76 \times 10^{-3}$ | adamw | $1 \times 10^{-3}$ | 0.44 | 256 | 16 | 98.33% | 0.0576 | 2 | 10 | 101.6 s (1.7 mnt) |
| **3** | stage_2+3 | $3.04 \times 10^{-5}$ | adam | 0.0 | 0.48 | 256 | 32 | 98.33% | 0.0749 | 38 | 40 | 840.4 s (14.0 mnt) |
| **4** | stage_3 | $3.77 \times 10^{-5}$ | adamw | $1 \times 10^{-4}$ | 0.37 | 0 | 32 | 98.33% | 0.0765 | 40 | 40 | 528.9 s (8.8 mnt) |
| **5** 🏆 | **beku** | **$1.67 \times 10^{-3}$** | **adam** | **0.0** | **0.43** | **256** | **32** | **98.67%** | **0.0530** | **25** | **33** | **294.1 s (4.9 mnt)** |

---

## 3. Analisis Model Terbaik (Trial 5)

* **Konfigurasi Terbaik:** **Trial 5** (dijalankan oleh Ilham), dipilih dari gabungan 10 trial di `trials.csv`.
* **Akurasi Validasi:** **98.67%** (296 dari 300 citra validasi benar), seri dengan trial 1, 6, dan 9.
* **Validation Loss:** **0.0530**, terendah di antara seluruh 10 trial, sehingga menjadi pemenang menurut aturan seri (val_loss terendah).
* **Titik Konvergensi Optimal:** Epoch 25 (early stopping berhenti di epoch 33).
* **Grafik Gradient Descent:** Tersimpan di berkas `history_convnext_tiny.png` (kurva trial 5).

### Pengamatan dari 10 Trial
1. **Backbone beku sudah cukup.** Trial terbaik memakai backbone yang dibekukan sepenuhnya. Membuka stage 3 atau stage 2+3 tidak menaikkan akurasi validation; semua trial berada di rentang 97.67%-98.67% (selisih paling besar 3 dari 300 gambar).
2. **Fine-tuning lebih mahal.** Trial yang membuka backbone butuh waktu lebih lama, misalnya trial 3 (stage 2+3) 840 detik dibanding 294 detik untuk trial 5.
3. **Dugaan (belum diuji):** fitur ImageNet dari ConvNeXtTiny sudah memisahkan 10 kelas SIMPLIcity dengan baik, sehingga dengan hanya 500 gambar latih, membuka lebih banyak layer menambah parameter tanpa memperbaiki hasil validation.

### Catatan untuk Trial 6 (terbaik di antara trial 6-10)
* **Konfigurasi:** stage_3, Adam $1.83 \times 10^{-4}$, dropout 0.43, head 256, batch 32. Val acc 98.67%, val loss 0.0580, epoch terbaik 15 (early stopping di epoch 23).
* **Dugaan faktor pendukung (belum diuji):**
  1. Membuka stage 3 (`convnext_tiny_stage_3_*`) memungkinkan filter tingkat tinggi menyesuaikan diri dengan objek SIMPLIcity, sementara layer bawah tetap memakai fitur ImageNet.
  2. Head 256 unit dengan dropout 0.43 memberi regularisasi pada data latih yang hanya 500 gambar.
  3. Kurva loss train dan validation turun mulus tanpa osilasi tajam. Ini konsisten dengan stabilitas Layer Normalization pada batch kecil, tetapi belum dibandingkan langsung dengan Batch Normalization.

