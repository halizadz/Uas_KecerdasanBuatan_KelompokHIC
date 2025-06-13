FROM python:3.9-slim

WORKDIR /app

# Install dependencies system
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt


COPY . .

# Gunakan Gunicorn dengan config produksi
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "api:app"]