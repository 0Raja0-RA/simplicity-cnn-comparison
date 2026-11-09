# Model-2: DenseNet121

Dokumentasi resmi untuk **Model-2 (`DenseNet121`)** pada eksperimen perbandingan 5 arsitektur CNN untuk klasifikasi 10 kategori gambar pada dataset SIMPLIcity (1.000 citra), Tugas Kelompok Neuro Computing (Minggu 11), PENS.

* **Anggota Pasangan:**
  * **Raja**: Bertanggung jawab mengeksekusi **Trial 1 s/d 5**
  * **Faruq**: Bertanggung jawab mengeksekusi **Trial 6 s/d 10**

---

## 1. Arsitektur & Ide Kunci Desain

* **Fungsi Keras:** `keras.applications.DenseNet121`
* **Bobot Awal:** Pretrained ImageNet (`weights="imagenet"`, `include_top=False`, `pooling="avg"`)
* **Input Resolusi:** $224 \times 224 \times 3$ (Letterbox padding dengan warna rata-rata ImageNet `(124, 116, 104)`)
* **Normalisasi:** `keras.applications.densenet.preprocess_input` dipasang secara **eksplisit** di dalam model (mode `"torch"`: piksel / 255, lalu dikurangi mean dan dibagi std ImageNet per channel). DenseNet di Keras **tidak** menormalisasi di dalam backbone-nya sendiri, sehingga pemanggilan `preprocess_input` wajib.
* **Jumlah Parameter Backbone:** $\approx 7.0 \text{ juta}$ parameter (tanpa head klasifikasi).
* **Ide Kunci Arsitektur (Huang et al., CVPR 2017):**
  1. **Dense Connection (Feature Reuse):** Di dalam satu dense block, setiap layer menerima gabungan (*concatenate*) feature map dari **semua** layer sebelumnya, bukan hanya layer tepat sebelumnya. Berbeda dengan ResNet yang *menjumlahkan* residual, DenseNet *menggabungkan* (concatenate), sehingga fitur dari layer awal tidak hilang dan bisa dipakai ulang secara langsung oleh layer-layer berikutnya.
  2. **Empat Dense Block dengan Transition Layer:** DenseNet121 memiliki 4 dense block (masing-masing 6, 12, 24, dan 16 layer, growth rate 32). Di antara tiap block terdapat *transition layer* (BatchNorm → konvolusi 1×1 → average pooling 2×2) yang memadatkan jumlah channel, menjaga agar dimensi tidak membengkak.
  3. **Efisiensi Parameter:** Karena setiap layer hanya menambah sedikit feature map baru (growth rate 32) dan memakai ulang fitur lama, DenseNet121 hanya membutuhkan $\approx 7$ juta parameter — jauh lebih sedikit dibanding ResNet50 ($\approx 25$ juta) — namun tetap kompetitif dalam akurasi.
  4. **Aliran Gradien yang Lancar:** Dense connection menyediakan jalur langsung (*short path*) dari loss ke layer-layer awal, sehingga gradien mengalir tanpa hambatan dan pelatihan konvergen lebih mudah, terutama pada dataset kecil seperti SIMPLIcity (500 citra training).

### Alasan Pemilihan & Kesesuaian dengan Dataset SIMPLIcity

1. **Ringan namun Kuat:** Dengan hanya $\approx 7$ juta parameter backbone, DenseNet121 menawarkan keseimbangan ideal untuk dataset kecil (500 citra training): cukup kompleks untuk menangkap pola visual 10 kelas yang beragam, namun tidak terlalu besar sehingga rawan *overfitting*.
2. **Feature Reuse Melawan Overfitting:** Mekanisme dense connection memaksa model memakai ulang fitur dari layer sebelumnya, berfungsi sebagai regularisasi implisit yang sangat berguna saat data terbatas.
3. **Kompatibilitas Transfer Learning:** Bobot pretrained ImageNet pada backbone DenseNet121 sudah mengandung representasi visual umum yang kaya. Dengan membekukan backbone dan hanya melatih head klasifikasi, model bisa beradaptasi ke SIMPLIcity dengan cepat (beberapa trial konvergen dalam 7–13 epoch).

---

## 2. Ruang Pencarian & Tabel 10 Trial Hyperparameter (`trials.csv`)

Seluruh 10 konfigurasi hyperparameter di-generate secara deterministik menggunakan **`SEED = 42`** di awal notebook sehingga urutan parameter di akun Raja dan Faruq **identik 100%**:
* **Sidik Konfigurasi:** `66ac9936e9e6`
* **Batas Maksimal Epoch:** 40 epoch
* **Early Stopping:** `monitor="val_loss"`, `patience=8`, `restore_best_weights=True`

| Trial | Buka Layer | Learning Rate | Optimizer | Weight Decay | Dropout | Unit Head | Batch Size | Penanggung Jawab |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | beku | $4.45 \times 10^{-4}$ | AdamW | $1 \times 10^{-4}$ | 0.41 | 256 | 16 | **Raja** |
| **2** | beku | $2.76 \times 10^{-3}$ | AdamW | $1 \times 10^{-3}$ | 0.44 | 256 | 16 | **Raja** |
| **3** | block3+4 | $4.70 \times 10^{-5}$ | Adam | 0.0 | 0.48 | 256 | 32 | **Raja** |
| **4** | block4 | $5.55 \times 10^{-5}$ | AdamW | $1 \times 10^{-4}$ | 0.37 | 0 (direct) | 32 | **Raja** |
| **5** | beku | $1.67 \times 10^{-3}$ | Adam | 0.0 | 0.43 | 256 | 32 | **Raja** |
| **6** | block4 | $1.87 \times 10^{-4}$ | Adam | 0.0 | 0.43 | 256 | 32 | **Faruq** |
| **7** | beku | $4.89 \times 10^{-4}$ | Adam | 0.0 | 0.25 | 0 (direct) | 32 | **Faruq** |
| **8** | block3+4 | $1.11 \times 10^{-4}$ | Adam | 0.0 | 0.30 | 256 | 32 | **Faruq** |
| **9** | block4 | $5.90 \times 10^{-5}$ | AdamW | $1 \times 10^{-5}$ | 0.24 | 256 | 16 | **Faruq** |
| **10** | beku | $9.76 \times 10^{-4}$ | Adam | 0.0 | 0.33 | 0 (direct) | 32 | **Faruq** |

> **Catatan Strategi Layer:**
> * `beku`: Seluruh backbone ImageNet dibekukan, hanya melatih head klasifikasi baru.
> * `block4`: Membuka dense block 4 teratas (layer berawalan `pool4_`, `conv5_`), transition layer 4 ikut terbuka.
> * `block3+4`: Membuka dense block 3 dan 4 (layer berawalan `pool3_`, `conv4_`, `pool4_`, `conv5_`).
> * **BatchNormalization selalu beku** di semua strategi: batch kecil (16/32) akan merusak statistik running mean/variance ImageNet.

---

## 3. Hasil 10 Trial dan Konfigurasi Terbaik

Gabungan `trials.csv` (Raja trial 1–5, Faruq trial 6–10). Semua trial dijalankan di GPU Kaggle Tesla T4, sidik `data_utils.py` `b3e0629783c5`, sidik konfigurasi `66ac9936e9e6`.

| Trial | Buka Layer | Val Acc | Val Loss | Best Epoch | Epoch Total | Durasi (s) | Dijalankan |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | beku | 0.9733 | 0.0872 | 13 | 21 | 103.2 | Raja |
| 2 | beku | 0.9733 | 0.0780 | 18 | 26 | 118.7 | Raja |
| 3 | block3+4 | 0.9600 | 0.1671 | 11 | 19 | 122.7 | Raja |
| 4 | block4 | 0.9700 | 0.0993 | 25 | 33 | 151.9 | Raja |
| **5** | **beku** | **0.9767** | **0.0863** | **7** | **15** | **72.4** | **Raja** |
| 6 | block4 | 0.9700 | 0.1137 | 6 | 14 | 80.7 | Faruq |
| 7 | beku | 0.9600 | 0.1283 | 39 | 40 | 167.4 | Faruq |
| 8 | block3+4 | 0.9667 | 0.1286 | 4 | 12 | 85.6 | Faruq |
| 9 | block4 | 0.9733 | 0.1202 | 9 | 17 | 97.7 | Faruq |
| 10 | beku | 0.9667 | 0.1002 | 39 | 40 | 172.5 | Faruq |

**Konfigurasi terbaik: Trial 5**, dipilih dari validation sebelum test. Trial 5 memiliki val_acc tertinggi (0.9767); tidak ada trial lain yang menyamai skor ini, sehingga tidak perlu tie-break berdasarkan val_loss.

| Parameter | Nilai |
| :--- | :--- |
| `buka_layer` | `beku` (seluruh backbone ImageNet dibekukan, hanya head yang dilatih) |
| `learning_rate` | $1.67 \times 10^{-3}$ |
| `optimizer` | Adam |
| `weight_decay` | 0.0 |
| `dropout` | 0.43 |
| `unit_head` | 256 (Dense ReLU lalu Dropout) |
| `batch_size` | 32 |
| Epoch terbaik | 7 dari 15 epoch (berhenti oleh early stopping) |

Kesepuluh trial berada di rentang val_acc 0.9600–0.9767. Selisih terbesarnya 0.0167, setara 5 dari 300 gambar validation. Menurut aturan kelompok, selisih di bawah 2% dianggap setara, jadi performa 10 trial relatif konsisten.

**Temuan dari tuning:**
- **Backbone beku unggul:** Empat trial terbaik (trial 5, 1, 2, 9) semuanya menggunakan backbone beku atau hanya membuka block teratas. Strategi `block3+4` justru menghasilkan val_acc terendah (trial 3: 0.9600), mengindikasikan bahwa fine-tuning terlalu dalam pada dataset sekecil SIMPLIcity menyebabkan overfitting.
- **Konvergensi cepat:** Trial 5 (pemenang) konvergen di epoch 7 — tercepat di antara 10 trial — menunjukkan bahwa fitur ImageNet DenseNet121 sudah sangat cocok untuk tugas ini tanpa perlu banyak iterasi.
- **Head 256 > tanpa head:** Trial tanpa hidden layer head (unit_head = 0) rata-rata menghasilkan val_acc lebih rendah (trial 4: 0.9700, trial 7: 0.9600, trial 10: 0.9667) dibanding yang menggunakan head Dense 256.

Evaluasi test akan dijalankan satu kali oleh Raja sebagai pemilik trial 5, setelah bagian ini di-commit. Hasilnya disimpan di `hasil_test.json` dan `confusion_densenet121.png`.

---

## 4. Hasil Evaluasi Test

> **Belum dijalankan.** Bagian ini akan diperbarui setelah evaluasi test satu kali di Kaggle dengan model `model-2_trial05.keras`.

---

## 5. Struktur Berkas di `model-2_densenet121/`

```text
model-2_densenet121/
├── model-2_densenet121.ipynb  # Notebook utama untuk Kaggle
├── trials_raja.csv            # Log hasil Trial 1-5 (Raja)
├── trials_faruq.csv           # Log hasil Trial 6-10 (Faruq)
├── history_raja.json          # Riwayat per epoch Trial 1-5
├── history_faruq.json         # Riwayat per epoch Trial 6-10
├── trials.csv                 # Tabel gabungan 10 trial (dihasilkan oleh sel gabung)
├── history_densenet121.png    # Grafik Loss & Akurasi Gradient Descent model terbaik
├── confusion_densenet121.png  # Confusion Matrix pada test set (belum)
├── hasil_test.json            # Ringkasan metrik akhir evaluasi test set (belum)
└── README.md                  # Dokumentasi ini
```

