# Use the official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first (caching optimization)
COPY requirements.txt .

# Pre-install CPU-only PyTorch to dramatically reduce image size and save build time
# (The default PyTorch Linux wheels contain massive CUDA binaries which are unnecessary here)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server using Uvicorn
# We bind to 0.0.0.0 and listen on the PORT environment variable (expected by Render/Railway)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
