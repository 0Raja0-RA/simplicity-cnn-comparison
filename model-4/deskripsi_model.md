# Deskripsi Arsitektur: ConvNeXtTiny

Dokumen ini menjelaskan karakteristik teknis dan arsitektur `ConvNeXtTiny` yang digunakan pada `model-4` untuk klasifikasi dataset SIMPLIcity (10 kategori), tugas kelompok Neuro Computing (Minggu 11), PENS.

---

## 1. Spesifikasi Teknis Keras

* **Fungsi Keras:** `keras.applications.ConvNeXtTiny`
* **Bobot Bawaan:** Pretrained ImageNet (`weights="imagenet"`, `include_top=False`, `pooling="avg"`)
* **Input Resolusi:** $224 \times 224 \times 3$ (Letterbox padding dengan warna ImageNet mean `(124, 116, 104)`)
* **Jumlah Parameter Total:** $\approx 28.6 \text{ juta}$ parameter ($\approx 27.8 \text{ juta}$ non-head)
* **Normalisasi:** Diolah secara **internal** oleh layer bawaan `convnext_tiny_prestem_normalization` (memetakan piksel float32 $0-255$ secara otomatis ke standar mean dan std ImageNet). Sesuai aturan nomor 5 panduan kelompok, tidak ditambahkan normalisasi eksternal di luar model.

---

## 2. Empat Pilar Arsitektur ConvNeXt (Liu et al., CVPR 2022)

ConvNeXt dirancang dengan mengambil arsitektur ResNet standar lalu memodifikasinya langkah demi langkah mengadopsi prinsip desain Vision Transformer (khususnya Swin Transformer), namun tetap mempertahankan operasi konvolusi murni tanpa mekanisme *self-attention*:

### a. Macro Design & Inverted Bottleneck (1:4 Expansion)
* Berbeda dengan ResNet tradisional yang menggunakan bottleneck standar (dimensi kanal dikompresi di tengah blok), ConvNeXt mengadopsi **Inverted Bottleneck** (dimensi kanal diperluas 4 kali lipat di dalam blok sebelum diproyeksikan kembali).
* Rasio komputasi stage dirancang $1:1:3:1$ (menyerupai proporsi Swin Transformer), mengalokasikan kapasitas representasi terbesar pada stage 2 dan 3.

### b. Depthwise Convolution $7 \times 7$ (Large Kernel)
* Operasi konvolusi spasial dipindahkan ke posisi awal blok dan diganti menjadi **Depthwise Convolution $7 \times 7$** (bukan standar $3 \times 3$).
* Kernel $7 \times 7$ memperluas *receptive field* lokal secara signifikan dalam satu langkah, memungkinkan model menangkap bentuk objek global (seperti bus, gajah, dan dinosaurus) secara utuh, meniru efek *local window attention* pada ViT.

### c. Layer Normalization (LN) Menggantikan Batch Normalization (BN)
* Batch Normalization (BN) memiliki kelemahan besar pada transfer learning dengan batch size kecil (16 atau 32), karena statistik mean dan varians mini-batch menjadi berisik (*noisy*).
* ConvNeXt mengganti seluruh BN dengan **Layer Normalization (LN)**, yang menormalisasi representasi fitur per-sampel melintasi seluruh kanal secara independen. Hasilnya adalah sinyal gradien yang sangat stabil dan kurva loss yang konvergen mulus.

### d. Fewer Activations & Normalizations dengan Aktivasi GELU
* Mengganti ReLU dengan fungsi aktivasi **GELU** (*Gaussian Error Linear Unit*).
* Jumlah layer aktivasi dan normalisasi dipangkas drastis: hanya dipasang satu layer LN dan satu layer GELU per blok, mencegah fenomena saturasi gradien dan degradasi informasi fitur pada layer-layer dalam.

---

## 3. Kesesuaian dengan Karakteristik SIMPLIcity

1. **Stabilitas pada Dataset Kecil (500 Train Citra):**
   Ketiadaan Batch Normalization membuat ConvNeXtTiny tidak terganggu oleh fluktuasi statistik batch kecil. Terbukti pada Trial 6 s/d 10, ConvNeXt tidak pernah mengalami lonjakan loss (*exploding loss*), berbeda dengan MobileNetV3 yang sensitif terhadap aktivasi non-linear.
2. **Kapasitas Representasi Tinggi (Akurasi > 98%):**
   Dengan 28 juta parameter yang telah dilatih pada ImageNet, ConvNeXtTiny mampu mengekstraksi tekstur natural (hewan, pemandangan, makanan) dengan presisi sangat tinggi, menghasilkan akurasi validasi puncak mencapai **98.67%**.

