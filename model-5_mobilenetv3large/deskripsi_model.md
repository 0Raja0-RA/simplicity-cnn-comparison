# Deskripsi Arsitektur: MobileNetV3Large

Dokumen ini menjelaskan karakteristik teknis dan arsitektur `MobileNetV3Large` yang digunakan pada `model-5` untuk klasifikasi dataset SIMPLIcity (10 kategori).

---

## 1. Spesifikasi Teknis Keras

* **Fungsi Keras:** `keras.applications.MobileNetV3Large`
* **Bobot Bawaan:** Pretrained ImageNet (`weights="imagenet"`)
* **Input Resolusi:** $224 \times 224 \times 3$
* **Jumlah Parameter Total (Backbone + Head SIMPLIcity):** $\approx 3.0 \text{ juta}$ parameter (3.005.962 pada $\alpha = 1.0$; $\approx 2.0 \text{ juta}$ jika $\alpha = 0.75$; $\approx 1.45 \text{ juta}$ jika varian `minimalistic=True`)
* **Normalisasi Input:** Rentang $[-1, 1]$ ditangani langsung di dalam model via layer internal `Rescaling(scale=1/127.5, offset=-1.0)` dengan menyetel parameter `include_preprocessing=True`.  
  *(Catatan Teknis Keras 3: Fungsi eksternal `keras.applications.mobilenet_v3.preprocess_input` di Keras 3 merupakan no-op passthrough. Oleh karena itu, normalisasi wajib diaktifkan via `include_preprocessing=True` agar bobot konvolusi ImageNet menerima rentang nilai $[-1, 1]$ yang benar).*

---

## 2. Tiga Pilar Arsitektur MobileNetV3Large

### a. Depthwise Separable Convolutions
Operasi konvolusi standar dipecah menjadi dua tahap:
1. **Depthwise Convolution:** Konvolusi $3 \times 3$ atau $5 \times 5$ diterapkan per-kanal secara terpisah untuk menyaring informasi spasial.
2. **Pointwise Convolution:** Konvolusi $1 \times 1$ untuk memetakan dan mengombinasikan fitur antar-kanal.
Hasilnya adalah pemangkasan beban komputasi FLOPs dan parameter hingga $\approx 85-90\%$ dibanding konvolusi standar dengan akurasi yang tetap terjaga.

### b. Squeeze-and-Excitation (SE) Attention Block
Blok SE terpasang pada modul bottleneck (*Inverted Residual Block*):
1. **Squeeze:** Melakukan *Global Average Pooling* pada fitur spasial untuk mengekstrak deskriptor kanal global.
2. **Excitation:** Melewatkannya ke dua layer Dense dengan rasio reduksi kanal (aktivasi ReLU dan Sigmoid) untuk mempelajari dependensi non-linear antar-kanal secara adaptif.
3. **Scale:** Mengalikan bobot *attention* yang dihasilkan ke *feature map* asli.
Mekanisme ini memungkinkan model secara adaptif memusatkan perhatian pada fitur yang paling diskriminatif (misalnya pola garis arsitektur, tekstur kulit gajah, atau warna kelopak bunga).

### c. Hard-Swish Activation ($h\text{-swish}$)
$$h\text{-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$
Aktivasi non-linear ini mengaproksimasi fungsi Swish ($x \cdot \sigma(x)$) tanpa memerlukan komputasi fungsi eksponensial ($e^{-x}$) yang mahal pada prosesor/perangkat bergerak, sehingga menjaga latensi inferensi tetap ultra-rendah ($\approx 1.47\text{ ms/gambar}$).

---

## 3. Kesesuaian dengan Karakteristik SIMPLIcity

1. **Efisiensi Kapasitas pada Dataset Kecil (500 Citra Train):**  
   Dengan kapasitas $\approx 3.0\text{M}$ parameter, MobileNetV3Large jauh lebih ramping dibanding arsitektur berbobot besar seperti ResNet152 atau DenseNet201, meminimalkan risiko *overfitting* sekaligus memaksimalkan generalisasi.
2. **Kekuatan Representasi Pretrained ImageNet:**  
   Bobot konvolusi terlatih ImageNet menyediakan ekstraksi fitur tingkat rendah (tepi, tekstur) hingga tingkat tinggi (bentuk objek) yang sangat matang. Dengan membekukan backbone (*feature extraction*), model mencapai akurasi test $98.00\%$ hanya dengan melatih 12.810 parameter pada layer Dense akhir.
