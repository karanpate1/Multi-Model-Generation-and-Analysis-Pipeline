# Use Python 3.12
FROM python:3.12-slim

# 1. Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# 2. Install Python Dependencies
RUN pip install --upgrade pip setuptools
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Application Code
COPY ./app ./app

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]