# Use Python 3.13 as base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create NLTK data directory and download required NLTK data
RUN mkdir -p /usr/local/share/nltk_data && \
    python -c "import nltk; \
    nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('wordnet', download_dir='/usr/local/share/nltk_data')"

# Copy source code
COPY . . 
RUN rm -f ./data/processed_spam_data.csv

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV FLASK_HOST=0.0.0.0

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]

# # Use Python 3.13 as base image
# FROM python:3.13-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies and download NLTK data
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-nltk \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Download NLTK data during build
# RUN python -m nltk.downloader punkt stopwords wordnet

# # Create necessary directories
# # RUN mkdir -p /app/data/new_data /app/data/processed /app/models /app/src

# # Copy source code
# COPY . . 
# RUN rm -f ./data/processed_spam_data.csv

# # Set environment variables
# ENV FLASK_APP=app.py
# ENV FLASK_ENV=production
# ENV FLASK_HOST=0.0.0.0

# # Expose port
# EXPOSE 5000

# # Command to run the application
# CMD ["python", "app.py"]

# # Dockerfile
# FROM python:3.13-slim

# # Set working directory
# WORKDIR /app

# # Copy requirements file
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Expose port
# EXPOSE 8000

# # Start the application
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

