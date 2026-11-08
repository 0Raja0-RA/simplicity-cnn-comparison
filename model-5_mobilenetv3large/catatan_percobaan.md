# Catatan Percobaan Model-5: MobileNetV3Large

Dokumen ini mencatat rincian desain eksperimen, perbedaan parameter pada setiap run (Trial 1 sampai 10), serta hipotesis yang diuji pada dataset SIMPLIcity.

---

## 1. Rincian Perbedaan Tiap Percobaan (Trial 1 - 10)

Setiap percobaan menguji variasi parameter dari fungsi Keras `keras.applications.MobileNetV3Large` (`alpha`, `minimalistic`, `pooling`) yang dipadukan dengan strategi pelatihan (`lr`, `optimizer`, `unfreeze_pct`, `batch_size`, `dropout`):

### Trial 1: Baseline Standar Feature Extractor
* **Parameter:** `alpha=1.0`, `minimalistic=False`, `pooling='avg'`, `lr=1e-3`, `optimizer='adam'`, `unfreeze_pct=0%`, `batch_size=32`, `dropout=0.2`
* **Fokus/Tujuan:** Menjadi titik acuan (*baseline*). Seluruh bobot backbone ImageNet dibekukan (hanya melatih dense head 10 kelas). Menggunakan konfigurasi default Keras: $\alpha=1.0$, SE block dan Hard-Swish aktif (`minimalistic=False`), serta GlobalAveragePooling.

### Trial 2: Penyesuaian Learning Rate & Peningkatan Dropout
* **Parameter:** `alpha=1.0`, `minimalistic=False`, `pooling='avg'`, `lr=5e-4`, `optimizer='adam'`, `unfreeze_pct=0%`, `batch_size=32`, `dropout=0.3`
* **Perbedaan dari Trial 1:** Learning rate diturunkan menjadi $5 \times 10^{-4}$ dan dropout dinaikkan menjadi $0.3$.
* **Fokus/Tujuan:** Menguji apakah langkah optimasi yang lebih halus dengan regularisasi dropout yang lebih ketat dapat menstabilkan kurva loss validasi dan mencegah overfitting pada head klasifikasi.

### Trial 3: Pengujian Parameter `alpha=0.75` (Width Multiplier)
* **Parameter:** `alpha=0.75`, `minimalistic=False`, `pooling='avg'`, `lr=1e-3`, `optimizer='adam'`, `unfreeze_pct=0%`, `batch_size=32`, `dropout=0.2`
* **Perbedaan dari Trial 1:** Mengubah nilai `alpha` dari `1.0` ke `0.75`.
* **Fokus/Tujuan:** Mengurangi lebar kanal filter di seluruh layer backbone sekitar 25% (parameter turun dari $\approx 4.2\text{M}$ menjadi $\approx 2.5\text{M}$). Menguji apakah arsitektur yang lebih ramping justru lebih tahan terhadap overfitting pada dataset kecil (500 citra train).

### Trial 4: Pengujian Parameter `minimalistic=True` (Tanpa SE & Hard-Swish)
* **Parameter:** `alpha=1.0`, `minimalistic=True`, `pooling='avg'`, `lr=1e-3`, `optimizer='adam'`, `unfreeze_pct=0%`, `batch_size=32`, `dropout=0.2`
* **Perbedaan dari Trial 1:** Mengaktifkan opsi `minimalistic=True`.
* **Fokus/Tujuan:** Menonaktifkan modul Squeeze-and-Excitation (SE) dan fungsi aktivasi Hard-Swish (menggunakan ReLU standar). Tujuannya untuk mengisolasi dan mengukur seberapa besar dampak modul attention SE dan Hard-Swish terhadap akurasi model pada SIMPLIcity.

### Trial 5: Pengujian Parameter `pooling='max'` (Global Max Pooling)
* **Parameter:** `alpha=1.0`, `minimalistic=False`, `pooling='max'`, `lr=5e-4`, `optimizer='adam'`, `unfreeze_pct=0%`, `batch_size=32`, `dropout=0.2`
* **Perbedaan dari Trial 2:** Mengubah pooling backbone dari `'avg'` ke `'max'`.
* **Fokus/Tujuan:** Menguji apakah mengambil aktivasi spasial paling dominan (peak features) lebih informatif untuk mengenali kategori citra tertentu (seperti keberadaan tekstur bunga atau objek dinosaurus) dibanding rata-rata global spasial.

### Trial 6: Pengujian Fine-Tuning Ringan (Unfreeze 20% Layer Teratas)
* **Parameter:** `alpha=1.0`, `minimalistic=False`, `pooling='avg'`, `lr=1e-4`, `optimizer='adam'`, `unfreeze_pct=20%`, `batch_size=32`, `dropout=0.3`
* **Perbedaan dari Trial 1 & 2:** Membuka 20% layer konvolusi teratas backbone untuk dilatih bersama head, dengan learning rate kecil ($1 \times 10^{-4}$).
* **Fokus/Tujuan:** Memberikan fleksibilitas pada representasi fitur level tinggi agar menyesuaikan dengan karakteristik visual spesifik SIMPLIcity tanpa merusak representasi low-level bawaan ImageNet.

### Trial 7: Eksplorasi Optimizer AdamW & Batch Size 16
* **Parameter:** `alpha=1.0`, `minimalistic=False`, `pooling='avg'`, `lr=3e-4`, `optimizer='adamw'`, `unfreeze_pct=0%`, `batch_size=16`, `dropout=0.2`
* **Perbedaan dari Trial 1:** Menggunakan optimizer AdamW (dengan decoupled weight decay) dan batch size lebih kecil (16).
* **Fokus/Tujuan:** Menguji apakah regularisasi bobot pada AdamW dikombinasikan dengan stochastic noise dari batch kecil (16) dapat membantu model mencapai generalisasi validasi yang lebih baik.

### Trial 8: Kombinasi `alpha=0.75` dengan Fine-Tuning 20%
* **Parameter:** `alpha=0.75`, `minimalistic=False`, `pooling='avg'`, `lr=1e-4`, `optimizer='adam'`, `unfreeze_pct=20%`, `batch_size=16`, `dropout=0.3`
* **Perbedaan:** Memadukan model ramping ($\alpha=0.75$) dengan fine-tuning 20% layer teratas pada batch size 16 dan dropout 0.3.
* **Fokus/Tujuan:** Menguji apakah arsitektur dengan kapasitas parameter lebih kecil ($\alpha=0.75$) lebih aman untuk di-fine-tune tanpa mengalami divergensi atau overfit pada data latih yang terbatas.

### Trial 9: Fine-Tuning 20% dengan Optimizer AdamW
* **Parameter:** `alpha=1.0`, `minimalistic=False`, `pooling='avg'`, `lr=2e-4`, `optimizer='adamw'`, `unfreeze_pct=20%`, `batch_size=16`, `dropout=0.3`
* **Perbedaan dari Trial 6 & 7:** Menerapkan fine-tuning 20% layer teratas namun dengan optimizer AdamW dan batch size 16.
* **Fokus/Tujuan:** Menguji stabilitas gradien saat fine-tuning dilakukan menggunakan mekanisme weight decay bawaan AdamW untuk mengontrol magnitudo perubahan bobot ImageNet.

### Trial 10: Fine-Tuning pada Varian `minimalistic=True`
* **Parameter:** `alpha=1.0`, `minimalistic=True`, `pooling='avg'`, `lr=1e-4`, `optimizer='adam'`, `unfreeze_pct=20%`, `batch_size=32`, `dropout=0.2`
* **Perbedaan dari Trial 4 & 6:** Menerapkan fine-tuning 20% pada varian arsitektur minimalistic.
* **Fokus/Tujuan:** Membandingkan kemampuan adaptasi fine-tuning varian tanpa SE block (Trial 10) terhadap varian dengan SE block (Trial 6).

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

## 3. Log Hasil Percobaan (Tersimpan di `trials.csv`)

*(Tabel ini otomatis terisi setiap kali notebook selesai menjalankan trial)*

| Trial | Val Acc | Val Loss | Best Epoch | Train Time (s) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | | | | | Menunggu run |
| 2 | | | | | Menunggu run |
| 3 | | | | | Menunggu run |
| 4 | | | | | Menunggu run |
| 5 | | | | | Menunggu run |
| 6 | | | | | Menunggu run |
| 7 | | | | | Menunggu run |
| 8 | | | | | Menunggu run |
| 9 | | | | | Menunggu run |
| 10 | | | | | Menunggu run |

