# About ResCat ML

## Deskripsi Aplikasi

**ResCat ML** adalah aplikasi berbasis Flask yang menggunakan machine learning untuk mendeteksi dan mengklasifikasi ciri-ciri wajah kucing. Aplikasi ini dirancang untuk membantu mengidentifikasi kondisi kesehatan kucing melalui analisis gambar wajah mereka.

## Fitur Utama

### 1. **Validasi Gambar Kucing**
- Menggunakan model MobileNetV3 untuk memverifikasi apakah gambar yang diupload adalah kucing atau bukan
- Mendeteksi probabilitas dengan threshold kustomisasi
- Mendukung klasifikasi berbagai jenis kucing (Persian, Siamese, Tabby, dll.)

### 2. **Deteksi Wajah Kucing**
- Menggunakan YOLO untuk mendeteksi dan melokalisir wajah kucing dalam gambar
- ROI (Region of Interest) selection otomatis dengan prioritas kepercayaan
- Mendukung multiple face detection dengan konfigurasi confidence level

### 3. **Landmark Detection**
- Mendeteksi titik-titik penting di wajah kucing (mata, telinga, mulut)
- Menggunakan model ONNX untuk deteksi bounding box dan landmark points
- Koordinat landmark detail untuk analisis lebih lanjut

### 4. **Klasifikasi Area Wajah**
- **Mata (Kiri & Kanan)**: Deteksi kondisi normal/abnormal pada mata kucing
- **Telinga (Kiri & Kanan)**: Analisis kesehatan telinga
- **Mulut**: Evaluasi kondisi mulut dan area sekitarnya
- Menggunakan EfficientNetB1 untuk setiap area dengan akurasi tinggi

### 5. **Background Removal**
- Otomatis menghapus background gambar menggunakan library rembg
- Sistem caching untuk optimasi performa
- Hash-based caching untuk menghindari pemrosesan ulang

### 6. **Grad-CAM Visualization**
- Menghasilkan heatmap visualisasi untuk menunjukkan area fokus model
- Membantu interpretabilitas hasil klasifikasi
- Berguna untuk debugging dan validasi model

## Teknologi yang Digunakan

- **Framework**: Flask (Python web framework)
- **ML Models**: 
  - MobileNetV3 (validasi kucing)
  - YOLO (deteksi wajah)
  - EfficientNetB1 (klasifikasi area)
  - ONNX Runtime (inferensi cepat)
- **Image Processing**: PIL/Pillow, NumPy
- **Background Removal**: rembg
- **Deep Learning**: TensorFlow, Keras

## Alur Kerja

1. **Upload Gambar** → User mengunggah foto kucing
2. **Validasi** → Sistem memverifikasi apakah gambar adalah kucing
3. **Background Removal** → Menghilangkan background untuk akurasi lebih baik
4. **Face Detection** → Mendeteksi posisi wajah kucing
5. **Landmark Detection** → Mengidentifikasi posisi mata, telinga, mulut
6. **Area Classification** → Mengklasifikasi setiap area (normal/abnormal)
7. **Hasil & Visualisasi** → Menampilkan hasil dengan confidence score dan Grad-CAM

## Use Case

Aplikasi ini cocok digunakan untuk:
- 🏥 **Veterinary Telemedicine**: Screening awal kondisi kesehatan kucing
- 📱 **Pet Care Apps**: Integrasi dalam aplikasi perawatan hewan
- 🔬 **Research**: Studi tentang penyakit kucing berbasis computer vision
- 👨‍⚕️ **Veterinary Consultation**: Alat bantu dokter hewan dalam diagnosis

## API Endpoints

- `GET /` - Web interface
- `POST /api/predict` - Prediksi lengkap dengan hasil klasifikasi
- `POST /api/validate` - Validasi apakah gambar adalah kucing
- `POST /api/detect-face` - Deteksi wajah kucing
- `GET /health` - Health check service

## Keunggulan

✅ **Multi-Model Architecture**: Menggunakan beberapa model spesialis untuk akurasi tinggi  
✅ **Explainable AI**: Grad-CAM visualization untuk transparansi hasil  
✅ **Optimized Performance**: Caching dan ONNX runtime untuk inferensi cepat  
✅ **Scalable**: Architecture modular yang mudah dikembangkan  
✅ **Production Ready**: Error handling, logging, dan health monitoring

---

**Version**: 1.0  
**License**: -  
**Contact**: -
