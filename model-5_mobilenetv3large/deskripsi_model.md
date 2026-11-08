# Deskripsi Arsitektur: MobileNetV3Large

Dokumen ini menjelaskan karakteristik teknis dan arsitektur `MobileNetV3Large` yang digunakan pada `model-5` untuk klasifikasi dataset SIMPLIcity (10 kategori).

---

## 1. Spesifikasi Teknis Keras

* **Fungsi Keras:** `keras.applications.MobileNetV3Large`
* **Bobot Bawaan:** Pretrained ImageNet (`weights="imagenet"`)
* **Input Resolusi:** $224 \times 224 \times 3$
* **Jumlah Parameter Total:** $\approx 4.2 \text{ juta}$ ($\approx 2.5 \text{ juta}$ jika $\alpha = 0.75$)
* **Normalisasi:** Skala piksel $[-1, 1]$ menggunakan `keras.applications.mobilenet_v3.preprocess_input`

---

## 2. Tiga Pilar Arsitektur MobileNetV3Large

### a. Depthwise Separable Convolutions
Operasi konvolusi standar dipecah menjadi dua tahap:
1. **Depthwise Convolution:** Konvolusi $3 \times 3$ atau $5 \times 5$ diterapkan per-kanal secara terpisah untuk menyaring informasi spasial.
2. **Pointwise Convolution:** Konvolusi $1 \times 1$ untuk memetakan dan mengkombinasikan fitur antar-kanal.
Hasilnya adalah pemangkasan beban komputasi FLOPs dan parameter hingga $\approx 85-90\%$ dibanding konvolusi standar.

### b. Squeeze-and-Excitation (SE) Attention Block
Blok SE terpasang pada bottleneck layer:
1. Melakukan *Global Average Pooling* pada fitur spasial (Squeeze).
2. Melewatkannya ke dua layer Dense dengan rasio reduksi kanal untuk mempelajari interaksi non-linear antar-kanal (Excitation).
3. Mengalikan bobot attention yang dihasilkan ke *feature map* asli.
Hal ini memperkuat kanal fitur yang paling relevan untuk membedakan kelas citra (misalnya membedakan tekstur gajah vs kuda vs makanan).

### c. Hard-Swish Activation ($h\text{-swish}$)
$$h\text{-swish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$
Aktivasi ini mengaproksimasi fungsi Swish ($x \cdot \sigma(x)$) tanpa komputasi fungsi eksponensial ($e^{-x}$) yang mahal pada prosesor, sehingga inferensi jauh lebih cepat.

---

## 3. Kesesuaian dengan Karakteristik SIMPLIcity

1. **Ukuran Dataset Kecil (500 Train Citra):**
   Model besar seperti ResNet152 atau DenseNet201 memiliki puluhan juta parameter yang mudah mengalami *overfitting* pada dataset kecil. MobileNetV3Large yang ringkas ($\approx 4.2\text{M}$ parameter) menawarkan regularisasi kapasitas yang alami (*inductive bias* kuat).
2. **Transfer Learning Efektif:**
   Fitur dasar ImageNet (garis, tekstur, bentuk) yang dipelajari MobileNetV3Large sangat kaya, sehingga hanya membutuhkan penyesuaian (*fine-tuning*) pada layer teratas untuk membedakan 10 kelas SIMPLIcity.

