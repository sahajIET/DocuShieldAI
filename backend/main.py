from fastapi import FastAPI, UploadFile, File, HTTPException,Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from fastapi.concurrency import run_in_threadpool # The key import for non-blocking execution
from typing import List
import os
import shutil
import tempfile
import logging
import sys
import pytesseract
import fitz
from services.pdf_redaction import redact_sensitive_info
from services.validation import post_process_and_validate_entities # PyMuPDF
# Import modularized core redaction logic from the new services package
from services.pdf_processing import extract_text_from_pdf_with_ocr
from services.entity_detection import (
    identify_entities_with_presidio,
    identify_sensitive_entities_regex,# spaCy function
    nlp,
    
)
# --- Logging Configuration ---
# Configure logging for the entire application, reading level from environment variable
log_level_str = os.getenv("DOCUSHIELD_LOG_LEVEL", "INFO").upper() # Get level from .env, default to INFO
log_level = getattr(logging, log_level_str, logging.INFO) # Convert string level to logging constant

logging.basicConfig(
    level=log_level, # Use the configurable log level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redaction_process.log", mode='w'), # Logs to a file
        logging.StreamHandler(sys.stdout)                     # Logs to the console
    ]
)
ALL_REDACTION_TYPES = [
    # Presidio types
    "PRESIDIO_CREDIT_CARD", "PRESIDIO_CRYPTO", "PRESIDIO_DATE_TIME", "PRESIDIO_DOMAIN_NAME",
    "PRESIDIO_EMAIL_ADDRESS", "PRESIDIO_IBAN_CODE", "PRESIDIO_IP_ADDRESS", "PRESIDIO_NRP",
    "PRESIDIO_LOCATION", "PRESIDIO_PERSON", "PRESIDIO_PHONE_NUMBER", "PRESIDIO_MEDICAL_LICENSE",
    "PRESIDIO_URL",
    # Regex types
    "REGEX_EMAIL_ADDRESS", "REGEX_PAN_ID", "REGEX_AADHAAR_ID", "REGEX_AMOUNT",
    "REGEX_IP_ADDRESS_V4", "REGEX_MAC_ADDRESS", "REGEX_POSTAL_ADDRESS", "REGEX_PREFIX_NAMES",
    "REGEX_SSN_US", "REGEX_AGE", "REGEX_BLOOD_GROUP", "REGEX_ALPHANUMERIC_ID_GENERIC", "REGEX_GENDER",
    # spaCy types
    # "SPACY_PERSON", "SPACY_NORP", "SPACY_ORG", "SPACY_GPE", "SPACY_LOC", "SPACY_DATE",
    # "SPACY_CARDINAL", "SPACY_MONEY", "SPACY_TIME", "SPACY_FAC",
]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="DocuShield AI Backend",
    description="API for AI-powered secure PDF redaction. Upload a PDF to get a redacted version.",
    version="0.4.0" # Version bump to reflect new validation logic
)

# --- CORS Middleware (Crucial for Frontend Integration) ---
# Get origins from environment variable, split by comma, strip whitespace
origins_str = os.getenv("DOCUSHIELD_CORS_ORIGINS", "http://localhost:5173,http://localhost:8000")
origins = [o.strip() for o in origins_str.split(',') if o.strip()] # Parse and filter empty strings

logging.info(f"CORS Origins configured: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the configurable origins list derived from .env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"], # Crucial for filename download
)

# --- Temporary File Storage Configuration ---
# Define a temporary directory for file storage, preferring environment variable
_temp_dir_from_env = os.getenv("DOCUSHIELD_TEMP_DIR")

if _temp_dir_from_env:
    TEMP_FILE_STORAGE_DIR = _temp_dir_from_env
    os.makedirs(TEMP_FILE_STORAGE_DIR, exist_ok=True)
    logging.info(f"Using custom temporary directory from .env: {TEMP_FILE_STORAGE_DIR}")
else:
    TEMP_FILE_STORAGE_DIR = tempfile.mkdtemp()
    logging.info(f"Using default system temporary directory: {TEMP_FILE_STORAGE_DIR}")

# --- Presidio Analyzer Initialization ---
# Initialize Presidio Analyzer Engine once when the app starts
try:
    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()
    logging.info("Presidio AnalyzerEngine initialized successfully.")
except ImportError:
    logging.error("Presidio Analyzer could not be imported. Please install it: pip install presidio-analyzer")
    analyzer = None

# --- Helper Functions ---

def cleanup_files(file_paths: list):
    """Safely deletes temporary files in a background task after the response is sent."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                logging.info(f"Cleaned up temporary file: {path}")
        except Exception as e:
            logging.error(f"Error during cleanup of file {path}: {e}")

def process_and_redact_pdf(input_path: str, output_path: str,redaction_types: List[str]) -> dict:
    """
    Synchronous helper function containing the core CPU-bound redaction logic.
    This is executed in a thread pool by `run_in_threadpool` to avoid blocking the server.

    Args:
        input_path: The file path of the original uploaded PDF.
        output_path: The file path where the redacted PDF should be saved.

    Returns:
        A dictionary with 'success' status and optional 'error_type'/'error_message'.
    """
    try:
        logging.info(f"=== Starting redaction process for {os.path.basename(input_path)} ===")

        # Step 1: Extract text and metadata from the PDF using OCR if necessary.
        logging.info("Step 1: Extracting text from PDF.")
        all_pages_data = extract_text_from_pdf_with_ocr(input_path)
        if not all_pages_data:
            logging.error(f"Failed to extract any text from PDF: {os.path.basename(input_path)}")
            return {"success": False, "error_type": "extraction_failed", "error_message": "Failed to extract text from PDF. The document might be empty or image-based with no readable text."}

        logging.info(f"Text extraction completed. Pages processed: {len(all_pages_data)}")

        # Step 2: Detect potential sensitive entities using multiple strategies.
        logging.info("Step 2: Detecting sensitive entities.")
        raw_detected_entities = []
        full_text_by_page = {p['page_num']: p['text'] for p in all_pages_data}

        for page_data in all_pages_data:
            page_num = page_data['page_num']
            text = page_data['text']
            is_ocr = page_data['is_ocr_page']
            ocr_details = page_data.get('ocr_word_details')

            # Use Presidio for advanced, context-aware NLP entity detection
            if analyzer:
                presidio_entities = identify_entities_with_presidio(analyzer, text, page_num, is_ocr, ocr_details,allowed_types=redaction_types)
                raw_detected_entities.extend(presidio_entities)

            # Use regex for pattern-based entity detection
            regex_entities = identify_sensitive_entities_regex(text, page_num, is_ocr, ocr_details,allowed_types=redaction_types)
            raw_detected_entities.extend(regex_entities)

        logging.info(f"Entity detection completed. Raw entities found: {len(raw_detected_entities)}")

        # Step 3: Post-process and validate entities to reduce false positives.
        logging.info("Step 3: Validating detected entities.")
        if nlp:
            validated_entities = post_process_and_validate_entities(raw_detected_entities, full_text_by_page, nlp, allowed_types=redaction_types)
            logging.info(f"Entity validation completed. Validated entities: {len(validated_entities)}")
        else:
            logging.warning("spaCy model (nlp) not loaded. Skipping entity validation.")
            validated_entities = raw_detected_entities

        # If no sensitive entities are found, no redaction is needed.
        if not validated_entities:
            logging.info(f"No sensitive entities found to redact in {os.path.basename(input_path)}.")
            # Copy the original file to the output path so the user gets their file back.
            shutil.copyfile(input_path, output_path)
            return {"success": True}

        # Step 4: Apply the redactions to the PDF.
        logging.info(f"Step 4: Applying redactions for {len(validated_entities)} entities.")
        redaction_result = redact_sensitive_info(input_path, validated_entities, output_path)

        if redaction_result:
            logging.info(f"=== Successfully completed redaction for {os.path.basename(input_path)} ===")
            return {"success": True}
        else:
            logging.error(f"Redaction function returned False for {os.path.basename(input_path)}")
            return {"success": False, "error_type": "redaction_failed", "error_message": "Failed to apply redactions to the PDF."}

    except Exception as e:
        # Log the full traceback for server-side debugging
        logging.error(f"An unexpected error occurred in the redaction process for {os.path.basename(input_path)}: {e}", exc_info=True)
        # Return a more specific error type and message for the main endpoint to handle
        if "memory" in str(e).lower() or "resource" in str(e).lower():
            error_message = "Processing failed due to insufficient server resources (e.g., very large PDF)."
            error_type = "resource_error"
        elif "ocr" in str(e).lower() or "tesseract" in str(e).lower() or "cv2" in str(e).lower():
            error_message = "Processing failed during OCR (e.g., malformed image data)."
            error_type = "ocr_processing_error"
        else:
            error_message = "An unexpected error occurred during PDF content processing."
            error_type = "processing_error"

        return {"success": False, "error_type": error_type, "error_message": error_message}

# --- API Endpoints ---
@app.get("/redaction-types", summary="Get Available Redaction Types", tags=["General"])
async def get_redaction_types():
    """Returns the master list of all supported redaction types."""
    return {"redaction_types": ALL_REDACTION_TYPES}

@app.get("/", summary="Root Endpoint", tags=["General"])
async def read_root():
    """Root endpoint with a welcome message and API documentation link."""
    return {"message": "Welcome to DocuShield AI API! Visit /docs for API documentation."}

@app.get("/health", summary="API Health Check", tags=["General"])
async def health_check():
    """Returns a status to indicate that the API is running and healthy."""
    return {"status": "ok", "message": "DocuShield AI API is healthy!"}

@app.post("/redact-pdf/",
          summary="Redact sensitive information from a PDF file",
          response_description="The redacted PDF file",
          tags=["Redaction"])
# MODIFIED at Line 172
async def redact_pdf(
    file: UploadFile = File(...),
    redaction_types: List[str] = Query(
        default=ALL_REDACTION_TYPES,
        title="Redaction Types",
        description="A list of entity types to redact. If not provided, all supported types will be redacted."
    )
):
    """
    Handles the PDF file upload, performs a quick validation for encryption,
    and then delegates the CPU-intensive redaction process to a background thread.

    - **Validates**: Checks for PDF file type.
    - **Pre-checks**: Immediately verifies if the PDF is password-protected or corrupted.
    - **Processes**: Redacts PII and other sensitive data.
    - **Returns**: A new, redacted PDF file.
    """
    # --- Step 1: Validate file type ---
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logging.error(f"File validation failed: Not a PDF. Filename: '{file.filename}'")
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    # --- Step 2: Define temporary file paths and a safe output filename ---
    original_basename = os.path.basename(file.filename)
    sanitized_name = os.path.splitext(original_basename)[0].strip('. ')
    output_filename = f"redacted_{sanitized_name}.pdf"

    input_path = os.path.join(TEMP_FILE_STORAGE_DIR, f"input_{original_basename}")
    output_path = os.path.join(TEMP_FILE_STORAGE_DIR, output_filename)

    try:
        # Save the uploaded file to the temporary directory
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Received file: '{file.filename}', saved to '{input_path}'")

        # --- Step 3: IMMEDIATE PRE-CHECK for password protection or corruption ---
        # This is the crucial check you requested. It happens before the heavy processing.
        try:
            doc = fitz.open(input_path)
            if doc.needs_pass:
                doc.close()
                logging.warning(f"Upload rejected: PDF '{file.filename}' is password-protected.")
                raise HTTPException(status_code=400, detail="The provided PDF is password-protected and cannot be processed.")
            doc.close()
            logging.info(f"PDF pre-check passed for '{file.filename}'.")
        except fitz.FileDataError as e:
            logging.warning(f"Upload rejected: PDF '{file.filename}' appears corrupted or invalid. Error: {e}")
            raise HTTPException(status_code=400, detail="The provided PDF is corrupted, invalid, or is not a standard PDF file.")

        # --- Step 4: Delegate the blocking, CPU-intensive function to the thread pool ---
        logging.info(f"Delegating redaction process for '{file.filename}' to thread pool.")
        result = await run_in_threadpool(
            process_and_redact_pdf,
            input_path=input_path,
            output_path=output_path,
            redaction_types=redaction_types
        )

        # --- Step 5: Handle the result from the threadpool function ---
        if not result.get("success"):
            error_message = result.get("error_message", "An unknown error occurred during processing.")
            logging.error(f"Redaction process failed for '{file.filename}': {error_message}")
            # The cleanup task will still run for input_path and output_path
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {error_message}")

        logging.info(f"Successfully processed '{file.filename}'. Returning '{output_filename}'.")

        # --- Step 6: Return the redacted file ---
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=output_filename,
            background=BackgroundTask(cleanup_files, [input_path, output_path])
        )

    except pytesseract.TesseractNotFoundError:
        logging.critical("Tesseract OCR engine not found. Please ensure Tesseract is installed and in your system's PATH.")
        # We must manually clean up here as the background task won't be set
        cleanup_files([input_path, output_path])
        raise HTTPException(status_code=500, detail="Server configuration error: OCR engine not found.")

    except HTTPException:
        # This block ensures that any HTTPException we raised (like for encrypted files)
        # is correctly propagated, after cleaning up the files.
        cleanup_files([input_path, output_path])
        raise

    except Exception as e:
        logging.error(f"An unexpected error occurred in the redact_pdf endpoint for '{file.filename}': {e}", exc_info=True)
        cleanup_files([input_path, output_path])
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")

