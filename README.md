# SmartGlass OCR API

## Inovasi untuk Meningkatkan Efisiensi Pembacaan Informasi oleh Tunanetra

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

SmartGlass OCR API adalah sistem berbasis REST API yang dirancang untuk membantu penyandang tunanetra mengakses informasi tertulis dengan lebih cepat dan efisien. Sistem ini mengkombinasikan teknologi Optical Character Recognition (OCR) dengan fitur peringkasan teks otomatis, yang memungkinkan pengguna untuk mendapatkan informasi penting dari gambar teks dengan cepat.

## Fitur Utama

- **OCR Canggih**: Konversi teks dari gambar atau PDF dengan dukungan multi-engine (Tesseract, EasyOCR, PaddleOCR)
- **Peringkasan Otomatis**: Menghasilkan ringkasan dari teks yang dikenali untuk akses informasi yang lebih cepat
- **Format Markdown**: Menyimpan hasil OCR dalam format Markdown yang terstruktur dan mudah dibaca
- **REST API**: Antarmuka API yang mudah digunakan untuk integrasi dengan aplikasi lain
- **Analisis Dokumen**: Deteksi otomatis struktur dokumen (tabel, formulir, paragraf, dll.)
- **Multi-bahasa**: Dukungan untuk bahasa Indonesia dan Inggris

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama
- **Flask**: Framework web untuk REST API
- **Tesseract OCR**: Engine OCR utama
- **EasyOCR & PaddleOCR**: Engine OCR alternatif untuk akurasi yang lebih baik
- **NLTK**: Pustaka NLP untuk pemrosesan teks dan peringkasan
- **OpenCV**: Pustaka pemrosesan gambar untuk pre-processing OCR

## Struktur Proyek

```
smartglass-ocr-api/
├── app/                         # Main application package
│   ├── __init__.py              # Application factory
│   ├── config.py                # Configuration settings
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   ├── ocr_processor.py     # OCR engine wrapper
│   │   └── markdown_formatter.py # Markdown generation
│   ├── api/                     # API endpoints
│       ├── __init__.py
│       ├── routes.py            # API routes
│       └── utils.py             # Helper functions
├── data/                        # Data storage
│   ├── uploads/                 # Temporary file uploads
│   └── markdown/                # Generated markdown files
├── lib/                         # External libraries
│   └── smartglass_ocr.py        # Original OCR engine
├── tests/                       # Test suite
│   └── test_api.py              # API tests
├── .gitignore                   # Git ignore file
├── requirements.txt             # Python dependencies
├── run.py                       # Application entry point
└── README.md                    # Project documentation
```

## Instalasi

### Prasyarat

- Python 3.8 atau lebih tinggi
- Tesseract OCR
- Poppler (untuk pemrosesan PDF)

### Langkah Instalasi

1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/smartglass-ocr-api.git
   cd smartglass-ocr-api
   ```

2. Buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   venv\Scripts\activate  # Windows
   ```

3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

4. Install dependensi sistem:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
   
   # MacOS
   brew install tesseract poppler
   
   # Windows - Download dan install Tesseract dari:
   # https://github.com/UB-Mannheim/tesseract/wiki
   ```

5. Jalankan aplikasi:
   ```bash
   python run.py
   ```

## Penggunaan API

### Endpoint API

- **GET /api/docs**: Dokumentasi API
- **POST /api/process**: Memproses file gambar atau PDF
- **GET /api/markdown**: Daftar semua file markdown
- **GET /api/markdown/\<filename\>**: Mendapatkan file markdown tertentu
- **GET /api/stats**: Statistik engine OCR

### Contoh Penggunaan

#### Memproses File Gambar

```bash
curl -X POST -F "file=@gambar.jpg" -F "language=ind+eng" http://localhost:5000/api/process
```

Response:
```json
{
  "status": "success",
  "message": "File processed successfully",
  "results": {
    "status": "success",
    "text": "Teks yang diekstrak dari gambar...",
    "confidence": 95.5,
    "summary": "Ringkasan dari teks...",
    "metadata": {
      "detected_language": "ind",
      "image_type": "document",
      "processing_time_ms": 1234.56
    }
  },
  "markdown_file": "gambar_1234567890.md",
  "markdown_url": "/api/markdown/gambar_1234567890.md"
}
```

#### Mendapatkan Daftar File Markdown

```bash
curl http://localhost:5000/api/markdown
```

Response:
```json
{
  "files": [
    {
      "filename": "gambar_1234567890.md",
      "created": "2024-05-03T14:30:45",
      "size": 2345,
      "url": "/api/markdown/gambar_1234567890.md"
    }
  ]
}
```

#### Mendapatkan File Markdown

```bash
curl -O http://localhost:5000/api/markdown/gambar_1234567890.md
```

## Format Output Markdown

File markdown yang dihasilkan mencakup:
- Metadata dokumen (judul, tanggal, status)
- Informasi pemrosesan (waktu, bahasa, kepercayaan)
- Ringkasan teks
- Wawasan kunci (key insights)
- Teks lengkap yang diekstrak
- Informasi terstruktur (jika terdeteksi)
- Statistik gambar

Contoh:
```markdown
---
title: OCR Results for gambar.jpg
date: 2024-05-03 14:30:45
status: success
language: ind
confidence: 95.5
image_type: document
engine: tesseract_otsu
---

# OCR Results: gambar.jpg
*Processed on: 2024-05-03 14:30:45*

## Processing Information

| Property | Value |
| -------- | ----- |
| Status | `success` |
| Processing Time | 1234.56 ms |
| Detected Language | ind |
| Image Type | document |
| OCR Engine | tesseract_otsu |
| Confidence | 95.50% |

## Summary

> Ringkasan teks yang diekstrak dari gambar. Berisi informasi penting dalam bentuk singkat.

## Key Insights

* Insight pertama yang penting
* Insight kedua yang relevan
* Insight ketiga sebagai tambahan informasi

## Extracted Text

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed et ligula eget nisl
ultricies efficitur. Donec auctor, nisl eget ultricies tincidunt, nisl nisl
aliquet nisl, nec aliquet nisl nisl eget nisl.

(text lengkap...)
```

## Konfigurasi

Konfigurasi aplikasi dapat disesuaikan melalui variabel lingkungan atau file `.env`:

- `UPLOAD_FOLDER`: Direktori untuk menyimpan file upload (default: "data/uploads")
- `MARKDOWN_FOLDER`: Direktori untuk menyimpan file markdown (default: "data/markdown")
- `PORT`: Port untuk menjalankan API (default: 5000)
- `DEBUG`: Mode debug (default: "false")
- `SECRET_KEY`: Secret key untuk aplikasi Flask (default: "smartglass-ocr-secret")

## Kontribusi

Kami sangat menghargai kontribusi Anda untuk pengembangan SmartGlass OCR API. Untuk berkontribusi:

1. Fork repositori
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

## Lisensi

Proyek ini dilisensikan di bawah lisensi MIT - lihat file `LICENSE` untuk detail.

## Penghargaan

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Flask](https://flask.palletsprojects.com/)
- [NLTK](https://www.nltk.org/)

## Informasi Kontak

Deswin Br Perangin Angin - 102022380449  
Universitas Telkom - [Fakultas Rekayasa Industri](https://fri.telkomuniversity.ac.id/)