# Gunakan image Python resmi
FROM python:3.10-slim

# Set direktori kerja di container
WORKDIR /app

# Salin file yang diperlukan ke dalam container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Tentukan port (Cloud Run default 8080)
ENV PORT 8080

# Jalankan aplikasi Flask
CMD ["python", "main.py"]
