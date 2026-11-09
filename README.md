# simplicity-cnn-comparison

Perbandingan 5 arsitektur CNN untuk klasifikasi 10 kategori gambar pada dataset SIMPLIcity (1.000 gambar). Tugas kelompok Neuro Computing (Minggu 11), PENS.

> Baca README ini sampai habis sebelum menulis kode. Aturan di bawah dibuat agar kelima model bisa dibandingkan secara adil. Kalau aturan dilanggar, hasil perbandingan tidak sah dan model harus dilatih ulang.

## 1. Tujuan

1. Membuat klasifikasi gambar 10 kategori pada SIMPLIcity.
2. Membuat model CNN terbaik. Kelompok (9 orang) dibagi menjadi 5 tim (4 pasangan dan 1 orang), tiap tim mengerjakan 1 arsitektur CNN yang berbeda.
3. Membuat grafik gradient descent (kurva loss dan akurasi) untuk training dan validation.
4. Membandingkan kelima model, lalu menganalisis faktor yang membuat satu model lebih unggul (cara layer disambung, jumlah parameter, dan sebagainya), bukan sekadar menyebut "lebih banyak layer".

## 2. Dataset

| Kategori | Kode | Kategori | Kode |
|---|---|---|---|
| People | 0-99 | Elephant | 500-599 |
| Beach | 100-199 | Flower | 600-699 |
| Building | 200-299 | Horse | 700-799 |
| Bus | 300-399 | Mountain | 800-899 |
| Dinosaur | 400-499 | Food | 900-999 |

Pembagian data (sama untuk semua pasangan, tidak boleh diubah):

| Split | Per kelas | Kode gambar | Total |
|---|---|---|---|
| Train | 50 | x00 - x49 | 500 |
| Validation | 30 | x50 - x79 | 300 |
| Test | 20 | x80 - x99 | 200 |

Dataset (~31 MB) **sudah di-commit ke repo** di folder `data/` (`train/`, `validation/`, `test/`), jadi cukup `git clone`. Isi `data/` **tidak boleh diubah, ditambah, atau dihapus**.

Struktur `data/` berupa berkas campur, **tanpa subfolder per kelas** (contoh: `data/train/0.jpg`). Label diturunkan dari nomor berkas: `kelas = nomor // 100` (0 = People, 1 = Beach, ..., 9 = Food). Ukuran gambar hanya dua: 384x256 (landscape, 80%) dan 256x384 (portrait, 20%), semuanya RGB. Detail eksplorasi ada di `common/eksplorasi_data.ipynb`.

Sumber: Wang, J. Z., Li, J., Wiederhold, G. "SIMPLIcity: Semantics-sensitive integrated matching for picture libraries." IEEE TPAMI, 23(9):947-963, 2001.

## 3. Struktur repo

```
simplicity-cnn-comparison/
├── README.md
├── .gitignore
├── data/                  <- di-commit, isinya tidak boleh diubah
│   ├── train/
│   ├── validation/
│   └── test/
├── common/
│   ├── data_utils.py           <- dikunci, dipakai lewat import
│   └── eksplorasi_data.ipynb
├── model-1/
├── model-2_densenet121/
├── model-3/
├── model-4/
├── model-5/
└── results/
```

### `common/` : kode bersama yang dikunci

- **Isi:** `data_utils.py`, satu-satunya kode pemrosesan data untuk seluruh pasangan.
- **Fungsinya:**
  - memverifikasi `data/` (jumlah per kelas, kebocoran antar-split, checksum `manifest.csv`);
  - memuat gambar train dan validation (`load_data()`), dengan label = nomor berkas // 100. Di Kaggle, sumbernya `simplicity_224.npz` dari dataset `yonobengkel12/simplicity-224-letterbox` (dipasang lewat *Add Input*, isinya dicek dengan checksum). Tanpa dataset itu, gambar dibaca dari JPEG di `data/`. Hasil kedua jalur identik;
  - mengubah ukuran gambar ke 224x224 dengan padding;
  - menyediakan layer augmentasi Keras (`get_augmentation()`), dipasang di awal model dan hanya aktif saat training;
  - menyediakan `load_test()` sebagai fungsi terpisah (lihat aturan test di bagian 5).
- **Keluaran:** piksel float32 0-255, label integer 0-9 (`sparse_categorical_crossentropy`). Normalisasi khusus tiap arsitektur (`preprocess_input` milik model) dipasang di dalam model masing-masing, bukan di `common/`.
- **Aturan:** dilarang mengubah, menyalin, atau menulis ulang berkas ini di notebook. Perubahan hanya dengan persetujuan seluruh kelompok, dan kalau berubah, kelima model wajib dilatih ulang.
- **Cara pakai:** mulai dari template notebook Kaggle yang dibagikan di grup WA. Sel setup-nya meng-clone repo (di Kaggle), lalu `from common import data_utils as du`. Untuk meminta bantuan AI, pakai template prompt yang juga dibagikan di grup.
- **Sidik resmi `data_utils.py`:** `b3e0629783c5` (commit `6b75918`). Setiap notebook mencetak sidik ini lewat `du.sidik_config()`. Kalau berbeda, notebook memakai versi yang tidak disepakati. Sidik ini **wajib diperbarui** setiap kali `data_utils.py` diubah.

### `model-1/` sampai `model-5/` : satu folder per pasangan

Isi tiap folder:

| Berkas | Fungsi |
|---|---|
| `model-N_<arsitektur>.ipynb` | Notebook utama: bangun model, tuning, evaluasi akhir |
| `trials.csv` | Log semua trial tuning (kolom di bagian 5) |
| `history_<arsitektur>.png` | Grafik gradient descent (loss dan akurasi train vs validation) model terbaik |
| `README.md` | Ringkasan: arsitektur, alasan pemilihan, konfigurasi terbaik, hasil, temuan |

Pasangan hanya boleh mengubah isi folder `model-N` miliknya. Folder pasangan lain dan `common/` tidak boleh disentuh.

### `results/` : perbandingan akhir

Diisi setelah semua pasangan selesai: tabel gabungan 5 model, confusion matrix, grafik perbandingan, dan analisis. Dikerjakan bersama.

## 4. Ketentuan pemilihan model

Model **tidak ditentukan dari awal**. Tiap pasangan memilih sendiri, dengan batasan berikut:

1. **Arsitektur CNN yang sudah dikenal** dan tersedia di Keras (`keras.applications` atau sumber resmi lain).
2. **Satu keluarga arsitektur hanya boleh dipakai satu pasangan.** Contoh: ResNet50 dan ResNet101 dianggap satu keluarga.
3. **Ide desain harus berbeda** supaya analisis "kenapa unggul" bermakna. Tulis di `model-N/README.md` ide kunci arsitektur yang dipilih (misalnya skip connection, dense connection, depthwise separable convolution, compound scaling).
4. **Bukan transformer** (ViT, Swin, dan sejenisnya). Tugas ini khusus CNN.
5. **Tidak memakai varian raksasa** (contoh: ResNet152, EfficientNetB5 ke atas). Model harus bisa dilatih di GPU Kaggle dengan waktu wajar. Menang hanya karena lebih besar bukan temuan yang menarik, dan lebih rawan overfitting pada data sekecil ini.
6. **Varian dikunci ke varian standar** (misalnya EfficientNetB0, DenseNet121). Eksplorasi varian lain boleh sebagai bonus di laporan, bukan di klasemen utama.
7. **Cara mengklaim model:** sebutkan di grup dan tulis di tabel di bawah. Siapa yang klaim lebih dulu berhak. Satu orang (koordinator) yang mengisi tabel supaya tidak terjadi konflik git.

| Folder | Pasangan | Arsitektur | Ide kunci | Status |
|---|---|---|---|---|
| model-1 | Juna, Fatih | ResNet50V2 | Skip connection | Tuning |
| model-2_densenet121 | Raja, Faruq | DenseNet121 | Dense connection | Tuning |
| model-3 | Tegar, Sony | EfficientNetV2B0 | Compound scaling + Fused-MBConv | Tuning |
| model-4 | Ilham, Arta | ConvNeXtTiny | CNN modern (kernel 7x7, LayerNorm) | Tuning |
| model-5 | Oka | MobileNetV3Large | Depthwise separable + SE | Tuning |

> **Keputusan:** bobot pretrained ImageNet (transfer learning) **boleh** dipakai. Semua pasangan memakai bobot pretrained ImageNet dari sumber resmi arsitekturnya.

## 5. Aturan main (berlaku untuk semua pasangan)

### Dikunci (sama untuk semua)

| Item | Ketentuan |
|---|---|
| Framework | Keras |
| Split data | Sesuai bagian 2 |
| Bobot awal | Pretrained ImageNet |
| Ukuran input | 224x224 |
| Cara resize | **Padding** (letterbox): perkecil sisi terpanjang menjadi 224 dengan rasio aspek tetap, lalu isi sisa ruang dengan warna rata-rata ImageNet RGB (124, 116, 104). Tidak memakai stretch (proporsi objek berubah) atau crop (informasi di tepi terpotong) |
| Augmentasi | Hanya train, dilakukan *on the fly*, ringan. Parameter dikunci: flip horizontal; rotasi maks 15 derajat; zoom in maks 20% (`(-0.2, 0.0)`); translasi maks 5%; brightness ±0.15; contrast ±0.15; area kosong diisi mode `reflect`. Dilarang: flip vertikal dan perubahan warna besar |
| Random seed | 42 untuk semua pasangan (`du.SEED` di `common/data_utils.py`), satu seed saja |
| Epoch maksimum dan early stopping | Maksimal 40 epoch per trial; `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)` |
| Normalisasi | `preprocess_input` milik arsitektur yang dipilih, dipasang di dalam model |
| Metode tuning | Random search |

### Bebas diubah per pasangan

Learning rate, strategi freeze/fine-tuning (persen layer teratas yang dibuka), optimizer, batch size, weight decay, dropout, dan jumlah unit pada head klasifikasi.

### Jatah tuning

- Satuan jatah = 1 trial, yaitu satu kali latih penuh (sampai early stopping) dengan satu kombinasi parameter.
- **Jatah: 10 trial per model**, sama untuk semua pasangan.
- Pemilihan konfigurasi terbaik hanya berdasarkan skor **validation**. Kalau seri, pilih val loss terendah.
- Semua trial dicatat di `trials.csv` dengan kolom: `trial, parameter..., val_acc, val_loss, best_epoch, train_time_sec`.
- Parameter yang hanya dimiliki satu arsitektur boleh di-tuning pasangan itu saja, tetapi tetap dalam jatah 10 trial.

### Aturan test set (paling penting)

1. Test set (`data/test`) **dilarang dipakai selama tuning**, termasuk untuk "sekadar mengintip".
2. Fungsi `load_test()` hanya dipanggil di **sel paling akhir** notebook.
3. Alur: tuning habis, pilih konfigurasi terbaik dari validation, **tulis konfigurasinya di `model-N/README.md` dan commit**, baru evaluasi test **satu kali** (akurasi, macro F1, confusion matrix).
4. Hasil test tidak boleh dipakai untuk memperbaiki apa pun. Kalau ada yang mengintip lalu mengubah parameter, hasil model itu dianggap tidak sah.
5. Protokol seed: **satu seed (42)**. Konfigurasi terbaik tidak dilatih ulang dengan seed lain.

## 6. Metrik pembanding akhir

Akurasi test, macro F1, jumlah parameter, ukuran berkas model, waktu latih, waktu inferensi per gambar, dan confusion matrix. Test set hanya 200 gambar (1 gambar = 0,5%), jadi selisih kecil bisa kebetulan. Karena hanya memakai satu seed, selisih akurasi test **kurang dari 2% (4 gambar)** dianggap **setara**.

## 7. Alur kerja git

1. `git pull` sebelum mulai bekerja.
2. Bekerja hanya di folder `model-N` milik sendiri.
3. Pesan commit diawali nama folder, contoh: `model-3: tambah trial 5-8`.
4. Jangan mengubah isi `data/` dan jangan commit berkas model (`.keras`, `.h5`, `.pt`, `.weights.h5`). Cukup `trials.csv`, grafik, dan notebook.
5. Bersihkan output notebook yang berukuran besar sebelum commit.

## 8. Urutan pengerjaan

1. Kelompok menyepakati: pretrained atau tidak, ukuran input, jenis augmentasi, jumlah trial, protokol seed.
2. Koordinator menulis `common/data_utils.py` dan push.
3. Pasangan klaim model dan mengisi tabel bagian 4.
4. Tiap pasangan tuning (10 trial), lalu membekukan konfigurasi terbaik.
5. Evaluasi test satu kali.
6. Bersama-sama mengisi `results/` dan menulis analisis.
