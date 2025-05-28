# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for model parameters
RUN mkdir -p param

# Copy application files
COPY app.py .
COPY param/* param/

# Expose port for Gradio
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"] 