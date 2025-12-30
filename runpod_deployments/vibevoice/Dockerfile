FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    pydub \
    torch \
    transformers \
    accelerate

# Clone and install VibeVoice
RUN git clone https://github.com/tejas-kale/VibeVoice.git /tmp/VibeVoice && \
    cd /tmp/VibeVoice && \
    pip install -e . && \
    cd / && \
    rm -rf /tmp/VibeVoice/.git

# Copy handler
COPY handler.py /app/handler.py

# Set the handler as the entrypoint
CMD ["python", "-u", "/app/handler.py"]
