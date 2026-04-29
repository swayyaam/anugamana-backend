# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies needed for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Production Final Image
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user for security
RUN useradd -m -u 1000 user

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R user:user /app

# Pre-download ONNX models to prevent slow cold-starts
RUN python -c "from transformers import AutoTokenizer; \
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2'); \
    ORTModelForFeatureExtraction.from_pretrained('Xenova/all-MiniLM-L6-v2', subfolder='onnx', file_name='model_quantized.onnx'); \
    AutoTokenizer.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2'); \
    ORTModelForSequenceClassification.from_pretrained('Xenova/ms-marco-MiniLM-L-6-v2', subfolder='onnx', file_name='model_quantized.onnx')"

USER user
ENV HOME=/home/user

# Expose the Gunicorn port
EXPOSE 8000

# Start Gunicorn with Uvicorn workers
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]