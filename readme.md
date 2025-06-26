# DocuShield AI: Secure PDF Redaction Service 🛡️📄

A full-stack web application designed to securely identify and redact sensitive information from PDF documents using OCR and NLP techniques.

## Project Overview

DocuShield AI automates the process of protecting privacy and ensuring compliance by removing Personally Identifiable Information (PII) and other confidential data from PDF documents. The application leverages a powerful combination of OCR (Optical Character Recognition) and NLP (Natural Language Processing) to provide intelligent, automated redaction capabilities.

This project demonstrates expertise in building robust, scalable web services, integrating advanced AI/ML capabilities, and deploying containerized applications to cloud environments.

## ✨ Key Features

- **PDF Upload & Preview**: Easy PDF document upload with in-browser preview capability
- **Intelligent Entity Detection**: Identifies sensitive data using three powerful strategies:
  - **Presidio Analyzer**: Context-aware PII detection (names, locations, credit cards, phone numbers)
  - **spaCy Named Entity Recognition (NER)**: General entity recognition (persons, organizations, geopolitical entities)
  - **Custom Regex Patterns**: Highly specific pattern-based data (Aadhaar IDs, PAN IDs, custom IDs)
- **Granular Redaction Control**: Fine-grained control over redaction types with selective entity filtering
- **Automated Redaction**: Permanent black redaction boxes over identified sensitive areas at pixel level (not layered on top), ensuring complete data removal that cannot be recovered through various extraction methods
- **OCR Integration**: Processes scanned or image-based PDFs using Tesseract OCR
- **AI-Powered Suggestions**: (Future/Planned) Gemini API integration for intelligent redaction recommendations
- **Responsive UI**: Clean, intuitive interface built with React and Tailwind CSS
- **Containerized Deployment**: Dockerized frontend and backend for scalable deployments

## 🏛️ Architecture

DocuShield AI follows a microservices architecture with distinct frontend and backend components:

### Frontend (Client-Side)
- Built with **React.js** for interactive user interface
- Handles file uploads, redaction type selection, and PDF preview
- Communicates with backend API for processing requests
- Manages progress display and redaction summaries

### Backend (API Service)
- Developed with **FastAPI** for high-performance API operations
- Handles PDF processing, OCR, entity detection, and redaction
- **Concurrent Request Handling**: Utilizes FastAPI's asynchronous capabilities with thread pool offloading for CPU-bound tasks
- Returns redacted PDFs with detailed redaction summaries

### Processing Pipeline

1. **Upload**: User uploads PDF to FastAPI backend
2. **Pre-check**: Validates PDF for password protection or corruption
3. **Text Extraction**: PyMuPDF extracts text; OCR processes image-based PDFs
4. **Entity Detection**: Multi-strategy analysis using Presidio, spaCy, and custom regex
5. **Validation & Consolidation**: Resolves overlapping detections for accurate redaction
6. **Redaction**: Applies black rectangles over sensitive areas at the Pixel level
7. **Download**: Returns redacted PDF to frontend

## 🚀 Technology Stack

### Frontend
- **React.js**: Component-based UI development
- **React Router DOM**: Declarative routing
- **Tailwind CSS**: Utility-first CSS framework
- **PDF.js**: In-browser PDF rendering
- **GSAP**: High-performance animations

### Backend (Python 3.10)
- **FastAPI**: Modern, high-performance web framework
- **Uvicorn**: ASGI server
- **PyMuPDF (fitz)**: PDF parsing, text extraction, and redaction
- **OpenCV**: Image processing for OCR
- **PyTesseract**: Tesseract OCR integration
- **Presidio-Analyzer**: PII detection and anonymization
- **spaCy**: Industrial-strength NLP with NER capabilities
- **Additional Libraries**: python-dotenv, aiofiles, python-multipart, numpy, pandas

### Deployment
- **Docker**: Containerization for both services
- **Render.com**: PaaS deployment platforms
- **Environment Consistency**: Portable, scalable container deployment

## 🧠 Deployment Challenges & Solutions

Successfully deployed on limited cloud resources (512MB RAM) through strategic optimizations:

### Memory Optimization Strategies
- **Optimized Base Images**: Switched to `python:3.10-slim-buster` for reduced memory footprint
- **Precise SpaCy Model Usage**: 
  - Configured Presidio to use compact `en_core_web_sm` model (~50MB) instead of `en_core_web_lg` (~587MB)
  - Explicit model configuration in `backend/services/entity_detection.py`
- **Production Readiness**: Removed `uvicorn --reload` flag to eliminate unnecessary memory overhead

These optimizations demonstrate practical problem-solving for real-world deployment constraints.

## ⚙️ Local Development Setup

### Prerequisites
- Git
- Python 3.10+
- Node.js (18+) & npm/Yarn
- Docker Desktop
- Tesseract OCR (for non-Docker development)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/sahajIET/DocuShieldAI.git
   cd docushield-ai
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   # Windows: .\venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Frontend Setup**
   ```bash
   cd ../frontend
   npm install
   npm start  # or npm run dev for Vite
   ```

### Docker Compose (Recommended)

```bash
# From project root
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173 / http://localhost:3000

## 📈 Future Enhancements

- **User Authentication & Authorization**: Login/signup with role-based access control
- **Asynchronous Processing**: Message queue integration for large documents
- **Enhanced UI/UX**: Drag-and-drop

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 Links

- [Live Demo](https://docushieldai-frontend.onrender.com/)
- [API Documentation](https://docushieldai-backend.onrender.com/docs)
- [Project Repository](https://github.com/sahajIET/DocuShieldAI)

---

**Built with ❤️ for secure document processing**