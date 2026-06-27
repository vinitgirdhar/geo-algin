# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install basic compiler toolchain and OpenCV libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code
COPY . /app/backend

# Set the PYTHONPATH so backend submodules import correctly
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["python", "backend/main.py"]
