# Model-1: ResNet50V2

Dokumentasi resmi **Model-1 (`ResNet50V2`)** pada eksperimen perbandingan 5 arsitektur CNN untuk klasifikasi 10 kategori gambar pada dataset SIMPLIcity (1.000 gambar). Tugas Kelompok Neuro Computing (Minggu 11), PENS.

- **Anggota pasangan:**
  * **Juna**: menjalankan **Trial 1 s/d 5** (beserta baseline)
  * **Fatih**: menjalankan **Trial 6 s/d 10** dan evaluasi test

---

## 1. Arsitektur & Ide Kunci Desain

- **Fungsi Keras:** `keras.applications.ResNet50V2`
- **Bobot awal:** pretrained ImageNet (`weights="imagenet"`, `include_top=False`, `pooling="avg"`)
- **Resolusi input:** $224 \times 224 \times 3$ (letterbox padding dengan warna rata-rata ImageNet `(124, 116, 104)`)
- **Normalisasi:** `keras.applications.resnet_v2.preprocess_input` (piksel diskalakan ke rentang `[-1, 1]`), dipasang di dalam model setelah layer augmentasi.
- **Jumlah parameter backbone:** 23.564.800 ($\approx 23{,}6$ juta), tanpa head klasifikasi. Dengan head `Dense(10)` (20.490 parameter), total model Trial 9 adalah 23.585.290.
- **Ide kunci arsitektur** (He et al., *Identity Mappings in Deep Residual Networks*, ECCV 2016; konsep residual berasal dari He et al., CVPR 2016):
  1. **Residual connection (skip connection):** koneksi *shortcut* menjumlahkan masukan blok dengan keluaran beberapa layer, sehingga layer-layer tersebut hanya perlu mempelajari fungsi *residual* $F(x) = H(x) - x$. Skema ini memudahkan optimisasi jaringan yang dalam dan mengatasi masalah degradasi (jaringan polos yang lebih dalam justru memiliki akurasi latih lebih rendah).
  2. **Pre-activation (V2):** berbeda dari ResNet V1, versi V2 menempatkan Batch Normalization dan ReLU *sebelum* konvolusi. Jalur *shortcut* menjadi identity murni (tanpa ReLU setelah penjumlahan), sehingga sinyal dan gradien dapat merambat langsung antar blok.
  3. **Blok bottleneck:** konvolusi 1×1 → 3×3 → 1×1. Konvolusi 1×1 pertama mengecilkan jumlah channel, konvolusi 3×3 bekerja pada channel yang lebih sedikit, dan konvolusi 1×1 terakhir mengembalikannya. Biaya komputasi lebih rendah daripada blok dua konvolusi 3×3 pada kedalaman yang sama.

### Alasan Pemilihan & Kesesuaian dengan Dataset SIMPLIcity

1. **Arsitektur mapan dengan kapasitas sedang.** ResNet50V2 banyak dipakai sebagai backbone *transfer learning* dan berukuran $\approx 23{,}6$ juta parameter, sehingga memenuhi batas "bukan varian raksasa" pada aturan pemilihan model.
2. **Ide desain berbeda dari model lain.** Skip connection berbeda dari *dense connection* (DenseNet), *compound scaling* (EfficientNetV2), kernel 7×7 + LayerNorm (ConvNeXt), dan *depthwise separable convolution* (MobileNetV3), sehingga analisis "mengapa satu model unggul" menjadi bermakna.
3. **Hipotesis untuk fine-tuning data kecil.** Jalur shortcut diharapkan membuat gradien tetap mengalir saat sebagian layer dibuka untuk *fine-tuning* pada 500 gambar latih. Ini hipotesis; eksperimen ini tidak memuat ablation yang menguji peran skip connection atau pre-activation secara terpisah.

---

## 2. Ruang Pencarian & Tabel 10 Trial Hyperparameter

10 kombinasi hyperparameter dihasilkan secara deterministik lewat *random search* (seed 42), sehingga kombinasinya **identik** di akun Juna dan Fatih.

- **Sidik `common/data_utils.py`:** `b3e0629783c5` (sidik kode data bersama, bukan sidik hyperparameter). Notebook memverifikasinya sebelum training.
- **Early stopping:** `monitor="val_loss"`, `patience=8`, `restore_best_weights=True`, diterapkan **pada setiap tahap** training.

**Skema training dua tahap** (sesuai notebook):

1. **Tahap 1:** backbone beku, hanya head yang dilatih dengan learning rate pada tabel. Maksimum 20 epoch jika trial memakai fine-tuning, atau 40 epoch jika backbone dibekukan penuh.
2. **Tahap 2** (hanya jika `Buka Layer` > 0): *N* layer teratas backbone dibuka dengan learning rate $= \text{lr}/10$, memakai sisa jatah epoch (maksimum total 40 epoch per trial). Jika tahap 2 tidak menurunkan *val_loss* dibanding akhir tahap 1, bobot tahap 1 dipakai.

| Trial | Buka Layer | Learning Rate (tahap 1) | Optimizer | Weight Decay | Dropout | Unit Head | Batch Size | Penanggung Jawab |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 60 layer | $1{,}0 \times 10^{-3}$ | Adam | 0,0 | 0,3 | 128 | 16 | **Juna** |
| **2** | beku (0 layer) | $1{,}0 \times 10^{-3}$ | Adam | 0,0 | 0,2 | 128 | 16 | **Juna** |
| **3** | 60 layer | $5{,}0 \times 10^{-4}$ | Adam | 0,0 | 0,2 | 256 | 16 | **Juna** |
| **4** | 30 layer | $5{,}0 \times 10^{-4}$ | Adam | 0,0 | 0,2 | 128 | 16 | **Juna** |
| **5** | 30 layer | $5{,}0 \times 10^{-4}$ | Adam | 0,0 | 0,2 | 0 (tanpa Dense) | 32 | **Juna** |
| **6** | 60 layer | $1{,}0 \times 10^{-3}$ | Adam | 0,0 | 0,3 | 256 | 32 | **Fatih** |
| **7** | 30 layer | $1{,}0 \times 10^{-3}$ | Adam | 0,0 | 0,3 | 128 | 16 | **Fatih** |
| **8** | beku (0 layer) | $1{,}0 \times 10^{-4}$ | Adam | 0,0 | 0,3 | 256 | 32 | **Fatih** |
| **9** | 60 layer | $1{,}0 \times 10^{-3}$ | Adam | 0,0 | 0,3 | 0 (tanpa Dense) | 16 | **Fatih** |
| **10** | beku (0 layer) | $1{,}0 \times 10^{-4}$ | Adam | 0,0 | 0,5 | 128 | 32 | **Fatih** |

> **Catatan:**
> * Optimizer (Adam) dan weight decay (0,0) **tidak divariasikan**; kolomnya dipertahankan agar sesuai format log kelompok.
> * `beku (0 layer)`: seluruh backbone dibekukan, hanya head klasifikasi yang dilatih.
> * `30 layer` / `60 layer`: 30 / 60 **objek layer Keras** terakhir pada backbone (termasuk layer BatchNormalization, ReLU, dan Add), bukan 30 / 60 blok konvolusi. Pada Trial 9, sekitar 17,2 juta parameter (73%) ikut dilatih karena layer-layer terakhir ResNet50V2 memuat sebagian besar parameter.
> * **BatchNormalization selalu dibekukan** pada semua strategi agar statistik ImageNet tetap stabil pada ukuran *batch* kecil.

---

## 3. Hasil 10 Trial dan Konfigurasi Terbaik

Gabungan `trials_juna.csv` (Trial 1–5) dan `trials_fatih.csv` (Trial 6–10), tersedia juga sebagai `trials.csv`. Semua trial dijalankan di Kaggle (Tesla T4). *Best Epoch* dihitung **kumulatif** (tahap 1 + tahap 2) berdasarkan *val_loss* terendah.

| Trial | Buka Layer | Val Acc | Val Loss | Best Epoch | Durasi (s) | Dijalankan |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 60 layer | 0,9433 | 0,1544 | 3 | 330,6 | Juna |
| 2 | beku | 0,9567 | 0,1422 | 4 | 181,9 | Juna |
| 3 | 60 layer | 0,9467 | 0,1589 | 4 | 374,0 | Juna |
| 4 | 30 layer | 0,9433 | 0,1480 | 4 | 395,9 | Juna |
| 5 | 30 layer | 0,9400 | 0,1584 | 20 | 368,1 | Juna |
| 6 | 60 layer | 0,9567 | 0,1849 | 6 | 342,2 | Fatih |
| 7 | 30 layer | 0,9433 | 0,1544 | 3 | 469,5 | Fatih |
| 8 | beku | 0,9533 | 0,1481 | 28 | 360,1 | Fatih |
| **9** | **60 layer** | **0,9633** | **0,1271** | **22** | **513,7** | **Fatih** |
| 10 | beku | 0,9467 | 0,1560 | 39 | 397,5 | Fatih |

Sebagai pembanding, **baseline** (backbone beku, tanpa dropout dan tanpa Dense, lr $10^{-3}$, batch 32; lihat `baseline_resnet50v2.csv`) mencapai *val_acc* 0,9500 dan *val_loss* 0,1837.

**Konfigurasi terbaik: Trial 9**, dipilih berdasarkan skor validation: *val_acc* tertinggi (0,9633) sekaligus *val_loss* terendah (0,1271).

| Parameter | Nilai |
| :--- | :--- |
| `buka_layer` | 60 (fine-tune pada 60 layer teratas backbone) |
| `learning_rate` | $10^{-3}$ (tahap 1), $10^{-4}$ (tahap 2) |
| `optimizer` | Adam |
| `dropout` | 0,3 |
| `unit_head` | 0 (tanpa Dense tambahan: Dropout lalu `Dense(10)` softmax) |
| `batch_size` | 16 |
| Epoch terbaik | 22 kumulatif (19 epoch tahap 1 + 3 epoch tahap 2); training berhenti di epoch ke-30 |

**Temuan dari tuning:**

- **Selisih antar-trial kecil.** *Val_acc* berada pada rentang 0,9400–0,9633, yaitu selisih 7 dari 300 gambar validation. Trial 9 unggul dari Trial 2 dan Trial 6 hanya 2 gambar, dan dari baseline 4 gambar. Dengan satu seed dan 300 gambar validation, perbedaan ini sebaiknya dibaca sebagai kecenderungan, bukan bukti yang pasti.
- **Fine-tuning parsial memberi keuntungan kecil.** Trial 9 (60 layer) memiliki *val_loss* terendah, tetapi Trial 2 (beku penuh) menyamai *val_acc* Trial 6 (0,9567) dengan waktu training paling singkat (181,9 s). Pada grafik Trial 9 (`history_best_resnet50v2_trial9.png`), *val_loss* terbaik tercapai tiga epoch setelah fine-tuning dimulai, lalu naik sementara *train loss* turun, tanda *overfitting*. *Early stopping* dengan `restore_best_weights` menahan model pada titik terbaik tersebut.
- **Tidak ada pola konsisten untuk ukuran head.** Dua trial tanpa Dense (`unit_head = 0`) menghasilkan *val_acc* tertinggi (Trial 9) sekaligus terendah (Trial 5). Karena *random search* mengubah beberapa hyperparameter sekaligus, pengaruh head tidak dapat dipisahkan dari faktor lain.
- **Batas epoch membatasi trial beku dengan lr kecil.** Trial 10 berjalan sampai batas 40 epoch dengan *best epoch* 39, dan Trial 8 mencapai *best epoch* 28, sehingga keduanya kemungkinan belum konvergen penuh. Performa strategi beku pada lr $10^{-4}$ bisa saja lebih tinggi jika jatah epoch lebih besar.
- **Model stabil terhadap hyperparameter.** Seluruh 10 trial mencapai *val_acc* $\ge$ 0,94 (Trial 5 tepat 0,9400).

Karena Trial 9 dijalankan oleh Fatih, **evaluasi test dijalankan satu kali oleh Fatih**.

---

## 4. Hasil Evaluasi Test (sekali, model Trial 9)

Dijalankan di GPU Kaggle (Tesla T4) dengan model `resnet50v2_trial9.keras` dari output run tuning, tanpa melatih ulang. Test set berisi 200 gambar (20 per kelas).

| Metrik | Nilai |
| :--- | :--- |
| Akurasi test | **0,9750** (195 dari 200 gambar benar) |
| Macro F1 | **0,9751** |
| Macro precision | 0,9774 |
| Macro recall | 0,9750 |
| Jumlah parameter total | 23.585.290 ($\approx 23{,}6$ juta) |
| Parameter dilatih | 17.201.162 ($\approx 17{,}2$ juta, 73% dari total) |
| Ukuran berkas model | 221,9 MB |
| Waktu inferensi | 11,28 ms per gambar |
| Waktu training Trial 9 | 513,7 detik (2 tahap, 30 epoch) |

**Catatan interpretasi:**

- Akurasi test (97,5%) konsisten dengan akurasi validation (96,3%). Dengan 200 gambar, interval kepercayaan 95% (Wilson) untuk akurasi test adalah sekitar 94,3%–98,9%. Menurut aturan kelompok, selisih akurasi test kurang dari 2% (4 gambar) antar-model dianggap setara.
- *Confusion matrix* (`confusion_resnet50v2.png`) mencatat 5 kesalahan: People → Beach dan People → Horse, Building → Beach dan Building → Elephant, serta Mountain → Beach. Tujuh kelas (Beach, Bus, Dinosaur, Elephant, Flower, Horse, Food) memiliki recall 100%. Beach menerima 3 salah prediksi sehingga *precision*-nya paling rendah (0,870).
- **Ukuran berkas** diukur dari berkas `.keras` yang disimpan lewat `model.save()` setelah `compile`, sehingga kemungkinan ikut memuat state optimizer Adam. Ukuran bobot saja sekitar 94 MB (23.585.290 parameter × 4 byte). Untuk perbandingan lima model, pastikan semua pasangan memakai cara pengukuran yang sama.
- *Parameter dilatih* adalah parameter berstatus *trainable* pada model akhir (setelah fine-tuning), bukan hanya parameter head.

Grafik *confusion matrix* tersimpan di `confusion_resnet50v2.png` dan ringkasan metrik di `hasil_test.json`.

---

## 5. Struktur Berkas di `model-1_resnet50v2/`

```text
model-1_resnet50v2/
├── model-1_resnet50v2_juna.ipynb       # Notebook baseline + Trial 1-5 (Juna)
├── trials.csv                          # Tabel gabungan 10 trial
├── trials_juna.csv                     # Log Trial 1-5 (Juna)
├── trials_fatih.csv                    # Log Trial 6-10 (Fatih)
├── detail_fatih.csv                    # Epoch per tahap, train acc/loss, val precision (Fatih)
├── baseline_resnet50v2.csv             # Hasil baseline
├── riwayat_baseline.json               # Riwayat per epoch baseline
├── history_resnet50v2.json             # Riwayat per epoch Trial 1-10
├── history_resnet50v2_baseline.png     # Grafik baseline
├── history_best_resnet50v2_trial9.png  # Grafik loss, akurasi, precision model terbaik (Trial 9)
├── png_history/                        # Grafik riwayat per trial (Trial 1-8 dan 10)
├── confusion_resnet50v2.png            # Confusion matrix pada test set
├── hasil_test.json                     # Ringkasan metrik evaluasi test
└── README.md                           # Dokumentasi ini
```
