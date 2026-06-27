# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

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
