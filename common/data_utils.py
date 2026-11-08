"""Pemrosesan data bersama untuk tugas kelompok CNN SIMPLIcity (5 model Keras).

Berkas ini adalah SATU-SATUNYA kode pemrosesan data untuk seluruh pasangan.
Jangan mengubahnya tanpa persetujuan seluruh kelompok; kalau berubah, kelima
model wajib dilatih ulang (README bagian 3 dan 5).

Ringkasan pipeline
------------------
- Sumber: dataset Kaggle simplicity_224.npz kalau terpasang. Gambarnya sudah
  di-letterbox dengan _letterbox() di bawah, jadi isinya identik dengan hasil
  dari JPEG. Kalau npz tidak ada, data/train, data/validation, data/test berupa
  berkas campur tanpa subfolder per kelas (contoh: data/train/0.jpg).
- Label: integer 0-9 = nomor berkas // 100, urutan sesuai CLASS_NAMES.
  Karena label berupa integer, pakai loss "sparse_categorical_crossentropy".
- Resize: padding (letterbox) ke IMG_SIZE. Sisi terpanjang diperkecil dengan
  rasio aspek tetap, sisa ruang diisi PAD_COLOR (rata-rata ImageNet).
- Output: piksel float32 rentang 0-255, BELUM dinormalisasi. Normalisasi
  khusus arsitektur (preprocess_input) dipasang di dalam model masing-masing.
- Augmentasi: TIDAK diterapkan di dataset. Pakai get_augmentation() sebagai
  layer pertama model; layer ini hanya aktif saat training, sehingga
  validation dan test tidak tersentuh.

Contoh pemakaian di notebook model-N/
--------------------------------------
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))        # root repo

    from tensorflow import keras
    from common.data_utils import (IMG_SIZE, cek_perangkat, load_data,
                                   get_augmentation, get_class_names)

    cek_perangkat()
    train_ds, val_ds = load_data(batch_size=32)

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = get_augmentation()(inputs)                    # aktif hanya saat training
    x = keras.applications.densenet.preprocess_input(x)
    base = keras.applications.DenseNet121(include_top=False, weights="imagenet",
                                          pooling="avg")
    x = base(x, training=False)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # ... tuning selesai, konfigurasi terbaik sudah ditulis di README model ...
    # SEL PALING AKHIR saja:
    # from common.data_utils import load_test, ambil_label
    # test_ds = load_test()

Cara pakai di Kaggle
--------------------
Pakai sel setup di template_kaggle.ipynb yang dibagikan di grup (Internet ON,
accelerator GPU). Intinya:
    !git clone https://github.com/0Raja0-RA/simplicity-cnn-comparison.git
    import sys
    sys.path.insert(0, "/kaggle/working/simplicity-cnn-comparison")
    from common import data_utils as du
    du.sidik_config()      # cocokkan dengan sidik resmi di README

Berkas ini DIKUNCI: jangan disalin, ditulis ulang, atau diubah di notebook.
Pakai hanya lewat import. du.sidik_config() membuktikan versi yang dipakai.

Dataset Kaggle: pasang dataset publik KAGGLE_DATASET lewat Add Input di panel
kanan notebook. load_data() dan load_test() lalu membaca simplicity_224.npz dari
/kaggle/input dan mencetak sumber datanya. Isi npz dicek dengan NPZ_SHA256
sebelum dipakai. Di luar Kaggle, berkas npz bisa ditunjuk lewat environment
variable SIMPLICITY_NPZ. Tanpa npz, gambar dibaca dari data/ hasil git clone
(folder lain bisa ditunjuk lewat SIMPLICITY_DATA_DIR), dan isi data/ tetap
diperiksa terhadap common/manifest.csv.

Menjalankan langsung
--------------------
    python common/data_utils.py                  # cek perangkat + verifikasi + uji 1 batch
    python common/data_utils.py --buat-manifest  # tulis common/manifest.csv
"""

from __future__ import annotations

import csv
import hashlib
import os
import platform
import re
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # verifikasi_data() dan buat_manifest() tetap bisa dipakai
    tf = None
    keras = None

# =============================================================================
# KONSTANTA (dikunci untuk semua pasangan)
# =============================================================================
SEED = 42                       # seed global: shuffle, augmentasi, bobot awal
IMG_SIZE = (224, 224)           # (tinggi, lebar) input model
PAD_COLOR = (124, 116, 104)     # warna padding RGB = rata-rata ImageNet (~0 setelah normalisasi)
BATCH_SIZE = 32                 # default; pasangan boleh mengubah lewat argumen

# Lokasi data: default <root repo>/data, bisa ditimpa env SIMPLICITY_DATA_DIR
DATA_DIR = Path(os.environ.get(
    "SIMPLICITY_DATA_DIR",
    Path(__file__).resolve().parent.parent / "data",
)).resolve()
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.csv"   # checksum SHA-256

# Dataset Kaggle berisi gambar yang sudah di-letterbox; dipakai otomatis kalau terpasang
KAGGLE_DATASET = "yonobengkel12/simplicity-224-letterbox"
NPZ_NAMA = "simplicity_224.npz"
NPZ_SHA256 = "0c140e73261690e18ef21347de34becd52e7c33dbd08b78518dac9eb5c567322"   # checksum isi npz

SPLITS = ("train", "validation", "test")
CLASS_NAMES = ("People", "Beach", "Building", "Bus", "Dinosaur",
               "Elephant", "Flower", "Horse", "Mountain", "Food")   # indeks = label
EXPECTED_PER_CLASS = {"train": 50, "validation": 30, "test": 20}   # gambar per kelas
KODE_PER_SPLIT = {"train": (0, 49), "validation": (50, 79), "test": (80, 99)}  # nomor % 100
EKSTENSI = (".jpg", ".jpeg", ".png")  # ekstensi gambar yang dibaca

# Parameter augmentasi (hanya train, lewat get_augmentation())
FLIP = "horizontal"             # tidak ada flip vertikal
ROTASI_DERAJAT = 15             # rotasi maksimal +/- 15 derajat
ZOOM = (-0.2, 0.0)              # Keras: negatif = zoom in; maks 20%, tanpa zoom out
TRANSLASI = 0.05                # geser maksimal 5% tinggi/lebar
BRIGHTNESS = 0.15               # +/- 15% dari rentang 0-255
CONTRAST = 0.15                 # faktor kontras 0.85-1.15
FILL_MODE = "reflect"           # pengisi area kosong setelah rotasi/zoom/geser

# Cache hasil pemeriksaan hash dalam satu sesi Python: {DATA_DIR: True}
_HASH_CACHE: dict[str, bool] = {}
_NPZ_CACHE: dict[str, dict] = {}    # isi npz yang sudah dimuat dalam satu sesi Python: {path: arrays}
_MAKS_DAFTAR = 10               # batas jumlah berkas yang disebut di pesan error


# =============================================================================
# Utilitas internal
# =============================================================================
def _perlu_tf():
    if tf is None:
        raise ImportError("TensorFlow belum terpasang; fungsi ini membutuhkan TensorFlow/Keras.")


def _kunci_urut(p: Path):
    """Urutkan berdasarkan angka pada nama berkas (2.jpg sebelum 10.jpg)."""
    return (0, int(p.stem), p.name) if p.stem.isdigit() else (1, 0, p.name)


def _daftar_gambar(split: str) -> list[Path]:
    folder = DATA_DIR / split
    if not folder.is_dir():
        raise ValueError(f"Folder {folder} tidak ditemukan. Periksa DATA_DIR / SIMPLICITY_DATA_DIR.")
    return sorted((p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in EKSTENSI), key=_kunci_urut)


def _batasi(daftar) -> str:
    daftar = list(daftar)
    teks = ", ".join(daftar[:_MAKS_DAFTAR])
    return teks + (f", ... (+{len(daftar) - _MAKS_DAFTAR} lagi)" if len(daftar) > _MAKS_DAFTAR else "")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _hash_data() -> dict[str, str]:
    """{path_relatif: sha256} untuk semua gambar di ketiga split."""
    return {f"{split}/{p.name}": _sha256(p) for split in SPLITS for p in _daftar_gambar(split)}


def _sha256_isi(arrays: dict) -> str:
    """Checksum isi npz: nama kunci, dtype, shape, dan byte tiap array, urut nama kunci."""
    h = hashlib.sha256()
    for kunci in sorted(arrays):
        a = np.ascontiguousarray(arrays[kunci])
        h.update(kunci.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _cari_npz():
    """Lokasi npz: env SIMPLICITY_NPZ, atau dataset yang terpasang di /kaggle/input. None kalau tidak ada."""
    if os.environ.get("SIMPLICITY_NPZ"):
        return Path(os.environ["SIMPLICITY_NPZ"])
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        hasil = sorted(kaggle.rglob(NPZ_NAMA))
        if hasil:
            return hasil[0]
    return None


def _muat_npz(path: Path) -> dict:
    """Muat npz sekali per sesi, lalu cocokkan checksum isinya dengan NPZ_SHA256."""
    kunci = str(path)
    if kunci not in _NPZ_CACHE:
        with np.load(path, allow_pickle=False) as f:
            isi = {k: f[k] for k in f.files}
        if _sha256_isi(isi) != NPZ_SHA256:
            raise ValueError(f"{path} bukan dataset npz yang disepakati kelompok (checksum berbeda).")
        _NPZ_CACHE[kunci] = isi
    return _NPZ_CACHE[kunci]


# =============================================================================
# Fungsi publik
# =============================================================================
def set_seed(seed: int = SEED) -> None:
    """Set seed Python, NumPy, dan TensorFlow sekaligus (keras.utils.set_random_seed)."""
    _perlu_tf()
    keras.utils.set_random_seed(seed)


def cek_perangkat() -> dict:
    """Deteksi GPU dan cetak ringkasannya. Tidak pernah menghentikan program.

    Returns:
        {"device": "GPU" atau "CPU", "n_gpu": int, "tf_version": str atau None}
    """
    if tf is None:
        print("TensorFlow belum terpasang. Hanya verifikasi_data() dan buat_manifest() yang bisa dipakai.")
        return {"device": "CPU", "n_gpu": 0, "tf_version": None}

    versi = tuple(int(x) for x in re.findall(r"\d+", tf.__version__)[:2])
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:  # GPU sudah diinisialisasi sebelumnya
                print("Memory growth tidak bisa diatur:", e)
            detail = tf.config.experimental.get_device_details(gpu)
            print("GPU:", detail.get("device_name", gpu.name))
        print("TensorFlow", tf.__version__)
        print("Memakai GPU")
        print("Kalau terjadi out-of-memory (OOM), turunkan batch size.")
    else:
        print("GPU tidak terdeteksi, memakai CPU")
        print("TensorFlow", tf.__version__)
        if platform.system() == "Windows" and versi >= (2, 11):
            print("Catatan: TensorFlow native di Windows tidak mendukung GPU sejak versi 2.11. "
                  "Opsi untuk memakai GPU: jalankan lewat WSL2.")
        print("Saran: eksplorasi dan uji data cukup di CPU, tetapi PELATIHAN model sebaiknya "
              "di Kaggle karena di CPU durasinya akan sangat lama.")
    return {"device": "GPU" if gpus else "CPU", "n_gpu": len(gpus), "tf_version": tf.__version__}


def buat_manifest() -> Path:
    """Hitung SHA-256 setiap gambar dan tulis common/manifest.csv (path_relatif, sha256).

    Baris diurutkan berdasarkan path_relatif supaya hasilnya deterministik.
    Jalankan sekali oleh koordinator, lalu commit manifest.csv.
    """
    hashes = _hash_data()
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path_relatif", "sha256"])
        for rel in sorted(hashes):
            w.writerow([rel, hashes[rel]])
    _HASH_CACHE.pop(str(DATA_DIR), None)
    print(f"Manifest ditulis: {MANIFEST_PATH} ({len(hashes)} berkas)")
    return MANIFEST_PATH


def _cek_manifest() -> None:
    kunci = str(DATA_DIR)
    if _HASH_CACHE.get(kunci):
        return
    if not MANIFEST_PATH.is_file():
        print(f"PERINGATAN: manifest belum dibuat ({MANIFEST_PATH.name}); pemeriksaan hash dilewati.")
        return
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        manifest = {row["path_relatif"]: row["sha256"] for row in csv.DictReader(f)}
    aktual = _hash_data()

    berbeda = sorted(k for k in manifest.keys() & aktual.keys() if manifest[k] != aktual[k])
    hilang = sorted(manifest.keys() - aktual.keys())
    tak_terdaftar = sorted(aktual.keys() - manifest.keys())
    if berbeda or hilang or tak_terdaftar:
        pesan = ["Isi data/ tidak sama dengan common/manifest.csv:"]
        if berbeda:
            pesan.append(f"- isi berbeda ({len(berbeda)}): {_batasi(berbeda)}")
        if hilang:
            pesan.append(f"- hilang ({len(hilang)}): {_batasi(hilang)}")
        if tak_terdaftar:
            pesan.append(f"- tidak terdaftar ({len(tak_terdaftar)}): {_batasi(tak_terdaftar)}")
        raise ValueError("\n".join(pesan))
    _HASH_CACHE[kunci] = True


def verifikasi_data(cek_hash: bool = True) -> dict:
    """Periksa data/ sebelum dipakai. Raise ValueError kalau ada yang tidak sesuai.

    Pemeriksaan:
        (a) tiap split berisi berkas campur, tanpa subfolder;
        (b) nama berkas berupa angka 0-999;
        (c) jumlah gambar per split per kelas sesuai EXPECTED_PER_CLASS;
        (d) nomor % 100 berada di rentang KODE_PER_SPLIT split-nya;
        (e) tidak ada nama berkas yang sama di dua split;
        (f) cek_hash=True: isi berkas sama dengan common/manifest.csv
            (dilewati dengan peringatan kalau manifest belum ada; hasilnya
            di-cache per sesi Python).

    Returns:
        {split: {nama_kelas: jumlah}}
    """
    galat = []
    jumlah = {s: {k: 0 for k in CLASS_NAMES} for s in SPLITS}
    split_per_nama: dict[str, list[str]] = {}

    for split in SPLITS:
        folder = DATA_DIR / split
        if not folder.is_dir():
            raise ValueError(f"Folder {folder} tidak ditemukan. Periksa DATA_DIR / SIMPLICITY_DATA_DIR.")
        subfolder = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if subfolder:
            galat.append(f"{split}: berisi subfolder {_batasi(subfolder)}; seharusnya berkas campur.")

        salah_nama, salah_rentang = [], []
        for p in _daftar_gambar(split):
            split_per_nama.setdefault(p.name, []).append(split)
            if not p.stem.isdigit() or not 0 <= int(p.stem) <= 999:
                salah_nama.append(p.name)
                continue
            kode = int(p.stem)
            jumlah[split][CLASS_NAMES[kode // 100]] += 1
            lo, hi = KODE_PER_SPLIT[split]
            if not lo <= kode % 100 <= hi:
                salah_rentang.append(p.name)
        if salah_nama:
            galat.append(f"{split}: nama berkas bukan angka 0-999: {_batasi(salah_nama)}")
        if salah_rentang:
            lo, hi = KODE_PER_SPLIT[split]
            galat.append(f"{split}: nomor di luar rentang x{lo:02d}-x{hi:02d}: {_batasi(salah_rentang)}")

        harap = EXPECTED_PER_CLASS[split]
        selisih = {k: n for k, n in jumlah[split].items() if n != harap}
        if selisih:
            galat.append(f"{split}: tiap kelas harus {harap} gambar, yang berbeda: {selisih}")

    bocor = sorted(f"{n} ({'/'.join(s)})" for n, s in split_per_nama.items() if len(s) > 1)
    if bocor:
        galat.append(f"Nama berkas muncul di lebih dari satu split: {_batasi(bocor)}")

    if galat:
        raise ValueError("Verifikasi data gagal:\n- " + "\n- ".join(galat))
    if cek_hash:
        _cek_manifest()
    return jumlah


def _letterbox(path):
    """Baca satu gambar lalu padding ke IMG_SIZE dengan PAD_COLOR. Output uint8 (H, W, 3)."""
    tinggi, lebar = IMG_SIZE
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear",
                          preserve_aspect_ratio=True, antialias=True)   # muat di dalam IMG_SIZE
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    atas, kiri = (tinggi - h) // 2, (lebar - w) // 2
    warna = tf.constant(PAD_COLOR, tf.float32)
    # tf.pad hanya mengisi 0, jadi geser dulu supaya 0 menjadi PAD_COLOR
    img = tf.pad(img - warna, [[atas, tinggi - h - atas], [kiri, lebar - w - kiri], [0, 0]]) + warna
    img = tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)
    return tf.ensure_shape(img, (tinggi, lebar, 3))


def _buat_dataset(split: str, batch_size: int, acak: bool, seed: int):
    npz = _cari_npz()
    if npz is not None:                           # dataset Kaggle: gambar sudah di-letterbox
        isi = _muat_npz(npz)
        ds = tf.data.Dataset.from_tensor_slices((isi[f"x_{split}"], isi[f"y_{split}"]))
        n = len(isi[f"y_{split}"])
    else:                                         # cadangan: baca JPEG dari data/
        paths = _daftar_gambar(split)
        labels = np.array([int(p.stem) // 100 for p in paths], dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], labels))
        ds = ds.map(lambda p, y: (_letterbox(p), y), num_parallel_calls=tf.data.AUTOTUNE)
        n = len(paths)
    ds = ds.cache()                               # decode + resize cukup sekali (uint8, hemat memori)
    if acak:
        ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_data(batch_size: int = BATCH_SIZE, seed: int = SEED):
    """Kembalikan (train_ds, val_ds) berisi (gambar float32 0-255, label int32 0-9).

    - Memanggil verifikasi_data() dan set_seed(seed) terlebih dahulu, lalu
      mencetak sumber data (npz dataset Kaggle atau JPEG di data/).
    - train_ds di-shuffle dengan seed (urutan berbeda tiap epoch, tetapi deterministik).
    - val_ds TIDAK di-shuffle, jadi ambil_label(val_ds) cocok dengan urutan prediksi.
    - Augmentasi TIDAK diterapkan di sini; pakai get_augmentation() di dalam model.
    - Parameter seed disediakan untuk protokol multi-seed; default SEED.
    """
    _perlu_tf()
    verifikasi_data()
    set_seed(seed)
    print("Sumber data:", sumber_data())
    train_ds = _buat_dataset("train", batch_size, acak=True, seed=seed)
    val_ds = _buat_dataset("validation", batch_size, acak=False, seed=seed)
    return train_ds, val_ds


def load_test(batch_size: int = BATCH_SIZE):
    """Kembalikan test_ds (gambar float32 0-255, label int32 0-9), TIDAK di-shuffle.

    PERINGATAN, ATURAN TEST SET (README bagian 5):
    - HANYA boleh dipanggil di SEL PALING AKHIR notebook model, SATU KALI.
    - Panggil hanya SETELAH tuning selesai dan konfigurasi terbaik sudah ditulis
      dan di-commit di model-N/README.md.
    - Hasil test tidak boleh dipakai untuk mengubah apa pun. Memanggil fungsi ini
      selama tuning, termasuk "sekadar mengintip", membuat hasil model tidak sah.
    """
    _perlu_tf()
    verifikasi_data()
    print("Sumber data:", sumber_data())
    return _buat_dataset("test", batch_size, acak=False, seed=SEED)


def get_augmentation():
    """Layer augmentasi Keras yang dikunci untuk semua pasangan.

    Pasang sebagai layer pertama model, SEBELUM preprocess_input. Input yang
    diharapkan: float 0-255. Layer ini otomatis hanya aktif saat training
    (training=True); saat evaluate/predict gambar dilewatkan tanpa perubahan.
    """
    _perlu_tf()
    L = keras.layers
    return keras.Sequential([
        L.RandomFlip(FLIP, seed=SEED),
        L.RandomRotation(ROTASI_DERAJAT / 360, fill_mode=FILL_MODE, seed=SEED),
        L.RandomZoom(height_factor=ZOOM, width_factor=ZOOM, fill_mode=FILL_MODE, seed=SEED),
        L.RandomTranslation(TRANSLASI, TRANSLASI, fill_mode=FILL_MODE, seed=SEED),
        L.RandomBrightness(BRIGHTNESS, value_range=(0, 255), seed=SEED),
        L.RandomContrast(CONTRAST, seed=SEED),
    ], name="augmentasi")


def ambil_label(ds) -> np.ndarray:
    """Kembalikan label asli (np.ndarray int) dalam urutan dataset, untuk confusion matrix.

    PERINGATAN: hanya benar kalau ds TIDAK di-shuffle (val_ds dari load_data()
    atau test_ds dari load_test()). Jangan dipakai pada train_ds, karena urutannya
    berubah setiap kali diiterasi dan tidak akan cocok dengan model.predict().
    """
    return np.concatenate([np.atleast_1d(y.numpy()) for _, y in ds])


def get_class_names() -> list[str]:
    """Salinan daftar nama kelas; indeks = label."""
    return list(CLASS_NAMES)


def sumber_data() -> str:
    """Sumber gambar untuk load_data() dan load_test(): berkas npz dataset Kaggle, atau folder data/ (JPEG)."""
    npz = _cari_npz()
    if npz is not None:
        return f"{npz} (npz dataset Kaggle)"
    return f"{DATA_DIR} (JPEG; dataset Kaggle {KAGGLE_DATASET} tidak terpasang)"


def sidik_config(cetak: bool = True) -> str:
    """SHA-256 isi berkas data_utils.py ini, sebagai bukti versi yang dipakai.

    Akhir baris dinormalisasi ke LF, sehingga hasilnya sama di Windows dan di
    Kaggle (Linux). Cocokkan 12 karakter pertamanya dengan sidik resmi di
    README bagian 3. Kalau berbeda, notebook memakai versi yang tidak disepakati.
    """
    isi = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    sidik = hashlib.sha256(isi).hexdigest()
    if cetak:
        print(f"Sidik data_utils.py: {sidik[:12]}  (lengkap: {sidik})")
    return sidik


# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utilitas data SIMPLIcity")
    parser.add_argument("--buat-manifest", action="store_true",
                        help="hitung SHA-256 semua gambar dan tulis common/manifest.csv")
    args = parser.parse_args()

    if args.buat_manifest:
        buat_manifest()
    else:
        sidik_config()
        info = cek_perangkat()
        print()
        jumlah = verifikasi_data()
        print(f"DATA_DIR: {DATA_DIR}")
        print(f"{'kelas':<10}" + "".join(f"{s:>12}" for s in SPLITS))
        for k in CLASS_NAMES:
            print(f"{k:<10}" + "".join(f"{jumlah[s][k]:>12}" for s in SPLITS))
        print(f"{'TOTAL':<10}" + "".join(f"{sum(jumlah[s].values()):>12}" for s in SPLITS))
        print("Verifikasi data: OK")

        if info["tf_version"] is not None:
            print("\nUji satu batch train:")
            train_ds, _ = load_data()
            x, y = next(iter(train_ds))
            x, y = x.numpy(), y.numpy()
            print(f"  shape gambar : {x.shape}, dtype {x.dtype}")
            print(f"  rentang      : {x.min():.1f} - {x.max():.1f}")
            print(f"  label        : {sorted(set(y.tolist()))} (dtype {y.dtype})")
            ada_padding = [(x[i] == np.array(PAD_COLOR, np.float32)).all(axis=-1).any()
                           for i in range(len(x))]
            assert x.shape == (BATCH_SIZE, *IMG_SIZE, 3), "shape batch tidak sesuai"
            assert x.dtype == np.float32, "dtype harus float32"
            assert 0.0 <= x.min() and x.max() <= 255.0, "rentang piksel harus 0-255"
            assert set(y.tolist()) <= set(range(10)), "label harus 0-9"
            assert all(ada_padding), "ada gambar tanpa area padding PAD_COLOR"
            print(f"  padding      : {sum(ada_padding)}/{len(x)} gambar memiliki area PAD_COLOR")
            print("Uji batch: OK")
