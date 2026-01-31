FROM python:3.9-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# ---------- System dependencies ----------
# - libgl1 / libglib2.0-0 → required by OpenCV
# - ffmpeg → required for video processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------- App directory ----------
WORKDIR /app

# ---------- Install Python deps first (cache-friendly) ----------
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=1000 -r requirements.txt


# ---------- Copy project ----------
COPY . .


RUN mkdir -p uploads static/audio
# ---------- Flask port ----------
EXPOSE 5000

# ---------- Run Flask ----------
CMD ["python", "app.py"]
