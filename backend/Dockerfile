FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH /app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install fastapi==0.111.0 && \
    pip install uvicorn[standard]==0.29.0 && \
    pip install python-dotenv==1.0.0 && \
    pip install PyMuPDF==1.23.25 && \
    pip install opencv-python-headless==4.9.0.80 && \
    pip install numpy==1.26.4 && \
    pip install pandas==2.2.2 && \
    pip install pytesseract==0.3.10 && \
    pip install Pillow==10.3.0 && \
    pip install presidio-analyzer==2.2.35 && \
    pip install spacy==3.7.4 && \
    pip install python-multipart==0.0.6 && \
    pip install aiofiles==23.2.1

# Download spaCy model
#RUN python -m spacy download en_core_web_sm

# Copy all application code
COPY . .

# IMPORTANT: Change working directory to backend folder
# This is where your main.py file is located
# WORKDIR /app/backend

EXPOSE 8000

# Now uvicorn can find main.py directly since we're in /app/backend/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]