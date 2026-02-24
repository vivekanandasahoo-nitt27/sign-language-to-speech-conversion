FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------- App directory ----------
WORKDIR /app

# ---------- Install deps (cache friendly) ----------
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# ---------- Copy project ----------
COPY . .

# ---------- Runtime dirs ----------
RUN mkdir -p uploads_v2 static/audio

# ---------- Flask port ----------
EXPOSE 5001

# ⭐ IMPORTANT — run as module (supports v2 imports)
CMD ["python", "-m", "v2.app_v2"]