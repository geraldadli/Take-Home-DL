# Studi Komparatif Model Generatif

Deep Learning (D6421), Sesi 06-07

## Isi

- [Notebook Colab](Generative_Models_Colab.ipynb) berisi eksperimen yang sudah dijalankan beserta metrik dan lima visualisasi.
- [Skrip PyTorch](src/) mencakup persiapan data, definisi model, pelatihan, evaluasi, analisis classifier, dan visualisasi.
- [requirements.txt](requirements.txt) mencantumkan dependensi untuk menjalankan skrip.

## Hasil utama

Fashion-MNIST dibagi ulang secara stratified 56.000/7.000/7.000, seed 42, batch training 128.
Split ini berbeda dari split resmi Fashion-MNIST. Training/evaluasi menggunakan Tesla T4.

| Model | Test MSE | PSNR (dB) | SSIM | FID | Latensi median (ms/batch 64) |
|---|---:|---:|---:|---:|---:|
| Vanilla AE | 0.008352 | 21.762 | 0.8306 | N/A | 0.982 |
| Sparse AE | 0.009538 | 21.153 | 0.8066 | N/A | 1.100 |
| VAE | 0.012803 | 19.782 | 0.7698 | 87.328 | 0.628 |
| Vanilla GAN | N/A | N/A | N/A | 105.887 | 1.008 |
| DCGAN | N/A | N/A | N/A | 19.791 | 3.357 |
| DDPM | N/A | N/A | N/A | 16.031 | 9395.755 |

Classifier: akurasi test 89.43% (syarat >=88%). Frozen decoder: MSE 0.017279,
dibandingkan AE 0.008352 dan transpose latent 0.279664.
FID memakai 7.000 real + 7.000 generated images, Inception-v3 2048 fitur.
Latensi AE/Sparse mencakup encode+decode; model lain generation. Hasil batch 64 pada T4
bukan bukti latensi batch 1 pada ponsel. Cakupan mode belum diukur secara kuantitatif.

## Menjalankan di Google Colab

1. Unggah `Generative_Models_Colab.ipynb` ke Colab.
2. Pilih runtime GPU dan jalankan sel secara berurutan.
3. Gunakan `PROFILE = "smoke"` untuk pemeriksaan cepat. Untuk hasil penuh gunakan
   `PROFILE = "full"` dan `EPOCH_OVERRIDE = {}`.
4. Jika Google Drive diaktifkan, output berada di `MyDrive/GenerativeModelsStudy`.
5. Jangan menggunakan hasil smoke sebagai hasil laporan. Mode full tetap perlu menjalankan
   evaluasi dan seluruh lima visualisasi setelah training.

Epoch full: AE/Sparse/classifier/frozen decoder 20; VAE/DDPM 30; GAN/DCGAN 40.
Checkpoint validation terpilih: AE 20, Sparse 20, VAE 29, DDPM 28, classifier 16,
frozen decoder 20. GAN/DCGAN menggunakan epoch final 40 yang telah ditentukan.
Perubahan konfigurasi menghasilkan folder run baru. Interupsi pada konfigurasi yang sama
dapat dilanjutkan dari checkpoint terakhir apabila checkpoint tersebut tersedia.
DDPM menggunakan 1.000 langkah reverse diffusion; FID DDPM dapat memerlukan waktu lama.

## Menjalankan skrip modular

Gunakan pasangan PyTorch/torchvision yang cocok untuk CUDA pada mesin tujuan.
Pertahankan pasangan bawaan Colab ketika memakai notebook.

```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.mlp_analysis
python -m src.figures
```

Default skrip adalah full. Untuk smoke, set environment variable `STUDY_PROFILE=smoke`.
Skrip menyimpan checkpoint, metrik, dan visualisasi di `generative_models_runs/`.

## Interpretasi

AE terbaik untuk rekonstruksi pada run ini, DDPM memiliki FID terendah, dan DCGAN memberi
kompromi kualitas/kecepatan. Fitur classifier masih menyimpan informasi rekonstruktif;
transpose bobot dan Gaussian diagonal tidak menyediakan generator yang otomatis valid.
Sampling classifier test-fit hanya diagnostik sesuai tugas; kontrol train-fit juga tersedia.
Satu seed, resolusi 28x28, serta jumlah parameter dan anggaran training berbeda membatasi
generalisasi. Baca laporan untuk jawaban Q1-Q6 dan batasan rekomendasi industri.

## Referensi

- Materi kuliah Sesi 06-07 dan panduan take-home D6421.
- VAE: https://arxiv.org/abs/1312.6114
- GAN: https://arxiv.org/abs/1406.2661
- DCGAN: https://arxiv.org/abs/1511.06434
- DDPM: https://arxiv.org/abs/2006.11239
- Inverse representations: https://arxiv.org/abs/1412.0035
- FID: https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
- PyTorch DCGAN: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
