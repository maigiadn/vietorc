FROM python:3.10-slim
WORKDIR /app

# Install OS dependencies for image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
