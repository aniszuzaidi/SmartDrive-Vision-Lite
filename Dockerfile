FROM python:3.10-slim

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application files
COPY . .

ENV ENABLE_SERVER_CAM=false
ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:7860", "web_app:app"]
