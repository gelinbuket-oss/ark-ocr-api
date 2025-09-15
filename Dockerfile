# Base image
FROM python:3.10-slim

# Tesseract ve bağımlılıkları yükle
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Çalışma klasörü
WORKDIR /app

# Gereken dosyaları kopyala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gunicorn ile çalıştır
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
