FROM python:3.10-slim

# Install system utilities needed by LightGBM and CatBoost compilers
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first to maximize Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your local files into the container workspace
COPY . .

# Expose ports for both our upcoming FastAPI backend (8000) and Streamlit web app (8501)
EXPOSE 8000
EXPOSE 8501

# Keep container awake by default
CMD ["bash"]