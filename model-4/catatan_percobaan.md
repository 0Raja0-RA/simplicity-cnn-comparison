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
| **6** 🏆 | **stage_3** | **$1.83 \times 10^{-4}$** | **adam** | **0.0** | **0.43** | **256** | **32** | **98.67%** | **0.0580** | **15** | **23** | **316.0 s (5.3 mnt)** |
| **7** | beku | $4.89 \times 10^{-4}$ | adam | 0.0 | 0.25 | 0 | 32 | 98.00% | 0.0791 | 40 | 40 | 364.1 s (6.1 mnt) |
| **8** | stage_2+3 | $9.31 \times 10^{-5}$ | adam | 0.0 | 0.30 | 256 | 32 | 98.00% | 0.0653 | 15 | 23 | 504.6 s (8.4 mnt) |
| **9** | stage_3 | $4.08 \times 10^{-5}$ | adamw | $1 \times 10^{-5}$ | 0.24 | 256 | 16 | 98.67% | 0.0627 | 39 | 40 | 534.2 s (8.9 mnt) |
| **10** | beku | $9.76 \times 10^{-4}$ | adam | 0.0 | 0.33 | 0 | 32 | 97.67% | 0.0665 | 40 | 40 | 365.6 s (6.1 mnt) |

---

## 3. Analisis Model Pemenang (Trial 6)

* **Konfigurasi Terbaik:** **Trial 6** (Dijalankan oleh Arta)
* **Akurasi Validasi:** **98.67%** (296 dari 300 citra validasi benar diklasifikasikan)
* **Validation Loss:** **0.0580** (paling rendah di antara seluruh trial)
* **Titik Konvergensi Optimal:** Epoch 15 (Early stopping berhenti di epoch 23)
* **Grafik Gradient Descent:** Tersimpan di berkas `history_convnext_tiny.png`.

### Faktor Keberhasilan Trial 6:
1. **Strategi Unfreeze Stage 3 Terpilih:**
   Membuka blok konvolusi teratas (`convnext_tiny_stage_3_*`) memungkinkan model menyesuaikan filter spasial $7 \times 7$ tingkat tinggi agar sensitif terhadap kontur spesifik objek SIMPLIcity (misal garis bus, fosil dinosaurus, dan tekstur makanan), sementara layer bawah tetap mempertahankan fitur tepi dasar ImageNet.
2. **Kombinasi Head 256 Unit + Dropout 0.43:**
   Dropout sebesar 0.43 terbukti memberikan regularisasi stokastik yang kuat, mencegah ko-adaptasi neuron pada data latih yang hanya berjumlah 500 sampel.
3. **Stabilitas Gradien Berkat Layer Normalization:**
   Kurva loss training dan validation menunjukkan penurunan eksponensial yang mulus tanpa osilasi tajam, membuktikan kekokohan Layer Normalization dibanding Batch Normalization pada ukuran batch 32.

