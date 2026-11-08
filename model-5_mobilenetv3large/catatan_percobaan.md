# Catatan Percobaan Model-5: MobileNetV3Large

Dokumen ini mencatat rincian desain eksperimen, perbedaan parameter pada setiap run (Trial 1 sampai 10), hasil evaluasi riil, serta analisis mendalam mengenai konfigurasi dan kelas yang unggul maupun yang kurang unggul pada dataset SIMPLIcity.

---

## 1. Rincian Perbedaan Tiap Percobaan (Trial 1 - 10)

Setiap percobaan menguji variasi parameter dari fungsi Keras `keras.applications.MobileNetV3Large` (`alpha`, `minimalistic`, `pooling`) yang dipadukan dengan strategi pelatihan (`lr`, `optimizer`, `unfreeze_pct`, `batch_size`, `dropout`):

* **Trial 1 (Baseline):** `alpha=1.0, minimalistic=False, pooling='avg', lr=1e-3, opt=adam, unfreeze=0%, bs=32, drop=0.2` (Fitur standar ImageNet, backbone beku).
* **Trial 2:** `lr=5e-4, drop=0.3` (Mengecilkan LR & menaikkan dropout untuk stabilisasi konvergensi).
* **Trial 3:** `alpha=0.75` (Width multiplier lebih kecil, menguji kapasitas model lebih ramping).
* **Trial 4:** `minimalistic=True` (Mematikan modul SE attention dan aktivasi Hard-Swish, menggunakan ReLU standar).
* **Trial 5:** `pooling='max'` (Menguji GlobalMaxPooling vs GlobalAveragePooling pada aktivasi spasial).
* **Trial 6:** `unfreeze=20%, lr=1e-4, drop=0.3` (Fine-tuning 20% layer konvolusi teratas dengan LR kecil).
* **Trial 7:** `opt=adamw, bs=16, lr=3e-4` (Menguji optimizer AdamW dengan decoupled weight decay & batch kecil).
* **Trial 8:** `alpha=0.75, unfreeze=20%, bs=16, lr=1e-4` (Kombinasi model ramping 0.75 dengan fine-tuning).
* **Trial 9:** `opt=adamw, unfreeze=20%, bs=16, lr=2e-4` (Fine-tuning 20% layer teratas dengan optimizer AdamW).
* **Trial 10:** `minimalistic=True, unfreeze=20%, lr=1e-4` (Fine-tuning pada varian minimalistic untuk pembanding Trial 6).

---

## 2. Tabel Ringkasan Rencana 10 Trial

| Trial | Alpha | Minimalistic | Pooling | Learning Rate | Optimizer | Unfreeze % | Batch Size | Dropout | Fokus Eksperimen |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `1.0` | `False` | `'avg'` | `1e-3` | `adam` | `0%` | `32` | `0.2` | Baseline feature extraction standard |
| **2** | `1.0` | `False` | `'avg'` | `5e-4` | `adam` | `0%` | `32` | `0.3` | Pengurangan LR & kenaikan dropout |
| **3** | `0.75`| `False` | `'avg'` | `1e-3` | `adam` | `0%` | `32` | `0.2` | Uji width multiplier lebih ramping |
| **4** | `1.0` | `True`  | `'avg'` | `1e-3` | `adam` | `0%` | `32` | `0.2` | Uji dampak ketiadaan modul SE & Hard-Swish |
| **5** | `1.0` | `False` | `'max'` | `5e-4` | `adam` | `0%` | `32` | `0.2` | Uji Global Max Pooling vs Average Pooling |
| **6** | `1.0` | `False` | `'avg'` | `1e-4` | `adam` | `20%`| `32` | `0.3` | Uji fine-tuning 20% layer teratas |
| **7** | `1.0` | `False` | `'avg'` | `3e-4` | `adamw`| `0%` | `16` | `0.2` | Uji optimizer AdamW & batch size 16 |
| **8** | `0.75`| `False` | `'avg'` | `1e-4` | `adam` | `20%`| `16` | `0.3` | Kombinasi alpha 0.75 + fine-tuning 20% |
| **9** | `1.0` | `False` | `'avg'` | `2e-4` | `adamw`| `20%`| `16` | `0.3` | Fine-tuning 20% dipadukan dengan AdamW |
| **10**| `1.0` | `True`  | `'avg'` | `1e-4` | `adam` | `20%`| `32` | `0.2` | Fine-tuning pada varian minimalistic |

---

## 3. Log Hasil Percobaan Riil (`trials.csv`)

Hasil eksekusi 10 trial pada GPU Kaggle (Tesla T4) dengan total durasi pelatihan $\approx 10.17$ menit:

| Trial | Alpha | Minimalistic | Pooling | LR | Optimizer | Unfreeze % | Batch | Dropout | Val Acc (%) | Val Loss | Best Epoch | Train Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.0 | False | avg | 0.001 | adam | 0 | 32 | 0.2 | 18.33% | 1,025,098.25 | 20 | 47.27 |
| **2** | 1.0 | False | avg | 0.0005 | adam | 0 | 32 | 0.3 | 20.00% | 970,760.75 | 35 | 58.35 |
| **3** | 0.75 | False | avg | 0.001 | adam | 0 | 32 | 0.2 | 16.00% | 521.99 | 38 | 54.34 |
| **4** | **1.0** | **True** | **avg** | **0.001** | **adam** | **0** | **32** | **0.2** | **47.67%** | **6.3143** | **40** | **48.49** |
| **5** | 1.0 | False | max | 0.0005 | adam | 0 | 32 | 0.2 | 21.67% | 2,216,675.75 | 40 | 61.46 |
| **6** | 1.0 | False | avg | 0.0001 | adam | 20 | 32 | 0.3 | 11.67% | 37,567.67 | 40 | 71.87 |
| **7** | 1.0 | False | avg | 0.0003 | adamw | 0 | 16 | 0.2 | 19.33% | 975,839.81 | 38 | 66.86 |
| **8** | 0.75 | False | avg | 0.0001 | adam | 20 | 16 | 0.3 | 13.00% | 11.75 | 40 | 71.54 |
| **9** | 1.0 | False | avg | 0.0002 | adamw | 20 | 16 | 0.3 | 11.00% | 1,364.33 | 40 | 77.96 |
| **10**| **1.0** | **True** | **avg** | **0.0001** | **adam** | **20** | **32** | **0.2** | **42.67%** | **1.6739** | **29** | **52.24** |

---

## 4. Konfigurasi Terbaik (Pemenang)

* **Trial Pemenang:** **Trial 4**
* **Validation Accuracy:** **47.67%** (`val_acc = 0.4767`)
* **Validation Loss:** **6.3143**
* **Spesifikasi Hyperparameter:**
  * `alpha`: 1.0
  * `minimalistic`: **True** (menggunakan aktivasi standar tanpa SE block)
  * `pooling`: 'avg' (GlobalAveragePooling)
  * `lr`: $1 \times 10^{-3}$ (Adam)
  * `unfreeze_pct`: 0% (Feature extraction dengan backbone beku)
  * `batch_size`: 32
  * `dropout`: 0.2
* **Grafik Gradient Descent:** Tersimpan di `history_mobilenetv3large.png` (memperlihatkan kurva konvergensi stabil dan penurunan loss dari epoch 1 hingga 40 tanpa divergensi).

---

## 5. Evaluasi Akhir Test Set (200 Citra)

Evaluasi dilakukan satu kali pada konfigurasi terbaik (Trial 4) menggunakan data test independen (20 citra per kelas):

* **Akurasi Test:** **49.00%** (98 dari 200 citra diklasifikasikan dengan benar)
* **Macro F1-Score:** **0.4868**
* **Macro Precision:** **0.5558**
* **Macro Recall:** **0.4900**
* **Jumlah Parameter Model Terbaik:** **1,447,218**
* **Waktu Inferensi:** **1.21 ms/gambar** (GPU Tesla T4)
* **Confusion Matrix:** Tersimpan di `confusion_matrix_mobilenetv3large.png`

### Laporan Performa per Kategori:

| Kategori | Precision | Recall | F1-Score | Status Performa |
| :--- | :---: | :---: | :---: | :--- |
| **Dinosaur** | **0.9091** | **1.0000** | **0.9524** | **Sangat Unggul** (20/20 benar terdeteksi) |
| **Elephant** | **0.8571** | **0.6000** | **0.7059** | **Sangat Unggul** (presisi tinggi) |
| **Flower** | **0.9091** | **0.5000** | **0.6452** | **Unggul** (presisi tinggi) |
| **Beach** | **0.4545** | **0.7500** | **0.5660** | **Cukup Baik** (recall tinggi) |
| **Bus** | **0.5625** | **0.4500** | **0.5000** | **Cukup** |
| **Horse** | **0.6000** | **0.3000** | **0.4000** | **Sedang** |
| **Food** | **0.2456** | **0.7000** | **0.3636** | **Kurang Unggul** (banyak false positive) |
| **People** | **0.3529** | **0.3000** | **0.3243** | **Kurang Unggul** |
| **Mountain** | **0.5000** | **0.2000** | **0.2857** | **Buruk** (recall sangat rendah) |
| **Building** | **0.1667** | **0.1000** | **0.1250** | **Paling Buruk** (hanya 2/20 benar) |

---

## 6. Analisis Konfigurasi & Kategori yang Kurang Unggul (Gagal)

Bagian ini menganalisis faktor penyebab kegagalan konfigurasi tertentu dan kelemahan model pada kategori citra spesifik:

### A. Analisis Trial yang Kurang Unggul / Gagal
1. **Kegagalan Fine-Tuning pada Varian Standar (Trial 6, 8, dan 9):**
   * Trial 9 (11.00%), Trial 6 (11.67%), dan Trial 8 (13.00%) mencatatkan akurasi terendah di seluruh eksperimen (nyaris setara dengan tebakan acak 10%).
   * **Penyebab:** Ketika 20% layer konvolusi teratas dibuka pada arsitektur dengan Squeeze-and-Excitation (`minimalistic=False`), terjadi fenomena *Catastrophic Forgetting*. Bobot ImageNet yang kaya fitur hancur karena gradien dari dataset kecil (500 citra train) mengubah bobot representasi secara agresif.
2. **Exploding Validation Loss pada Varian Non-Minimalistic (Trial 1, 2, 5, 7):**
   * Pada Trial 1, 2, 5, dan 7, nilai `val_loss` meledak hingga ratusan ribu hingga jutaan ($970.000 - 2.200.000$).
   * **Penyebab:** Fungsi aktivasi Sigmoid pada modul SE attention dan Hard-Swish mengalami saturasi ekstrem ketika menghadapi citra di luar distribusi ImageNet tanpa warm-up learning rate yang panjang. Akibatnya, logits keluaran bernilai ekstrem dan menghasilkan penalti cross-entropy yang masif.
3. **Kelemahan Global Max Pooling (Trial 5):**
   * Global Max Pooling (`pooling='max'`) hanya mencapai akurasi 21.67% dengan loss meledak ke 2.216.675.
   * **Penyebab:** Max pooling hanya mengambil 1 nilai aktivasi paling tajam per kanal spasial. Pada citra dengan latar belakang kompleks, max pooling cenderung menangkap noise latar belakang ketimbang konteks objek utama.

---

### B. Analisis Kategori Citra yang Paling Lemah
1. **Building (Paling Lemah - F1: 0.1250):**
   * Dari 20 citra uji Building, hanya **2 citra** yang berhasil ditebak dengan benar. Sebagian besar salah diklasifikasikan sebagai **Beach (6 citra)** dan **Food (5 citra)**.
   * **Faktor Penyebab:** Gedung arsitektur kuno dan modern di SIMPLIcity memiliki variasi tekstur vertikal/horizontal yang sering berbaur dengan latar belakang langit biru cerah dan tanah/pantai, membuat representasi spasial model bingung membedakan batas gedung dengan pemandangan alam pantai.
2. **Mountain (Sangat Lemah - F1: 0.2857):**
   * Hanya **4 dari 20 citra** yang berhasil ditebak (Recall 20%). Sebanyak 5 citra salah ditebak sebagai Food dan 5 citra sebagai Building.
   * **Faktor Penyebab:** Bentuk lereng batu dan puncak bersalju gunung memiliki gradien warna abu-abu/putih/cokelat yang menyerupai fasad beton gedung perkotaan atau piring saji makanan.
3. **Food (False Positive Terlalu Tinggi - Precision 24.56%):**
   * Model memiliki kecenderungan bias yang sangat kuat untuk memprediksi kelas Food. Dari 200 citra uji, model menebak sebanyak **57 kali** sebagai Food, padahal hanya 14 yang benar-benar Food.
   * Citra dari kelas People (12 citra), Bus (9 citra), Mountain (5 citra), dan Building (5 citra) banyak terserap ke kelas Food karena warna-warni kontras tinggi pada makanan serupa dengan pernak-pernik pakaian manusia dan warna bus di perkotaan.
