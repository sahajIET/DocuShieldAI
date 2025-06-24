# [Your imports remain the same]
import fitz
import pytesseract
from PIL import Image
import io
import spacy
import os
import sys
import pandas as pd
import numpy as np
import cv2
import re
import logging

# --- Logger Setup ---
# This is correctly set up. No changes needed here.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redaction_process.log", mode='w'), # 'w' overwrites the log each run
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Load spaCy Model ---
try:
    nlp = spacy.load("en_core_web_lg")
    # HIGHLIGHT: Replaced print with logging
    logging.info("spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    # HIGHLIGHT: Replaced print with logging
    logging.error(f"Error loading spaCy model: {e}", exc_info=True)
    logging.critical("spaCy model failed to load. The application cannot proceed.")
    nlp = None

# --- OCR PROFILES and ANALYSIS (No changes needed here) ---
# This part of your code is already excellent.

OCR_PROFILES = {
    "standard_scan": {"dpi": 400, "psm": 6, "denoise": True, "thresh_block_size": 29, "thresh_c": 5, "description": "Good for average quality office scans."},
    "high_quality_digital": {"dpi": 400, "psm": 6, "denoise": False, "thresh_block_size": 51, "thresh_c": 10, "description": "Best for clean, high-contrast, digitally-born documents."},
    "noisy_or_low_contrast": {"dpi": 450, "psm": 6, "denoise": True, "thresh_block_size": 15, "thresh_c": 4, "description": "Optimized for faxes, photos of documents, or poor quality scans."}
}

def analyze_image_and_select_profile(image):
    mean, std_dev = cv2.meanStdDev(image)
    contrast = std_dev[0][0]
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    LOW_CONTRAST_THRESHOLD = 55.0
    HIGH_CONTRAST_THRESHOLD = 80.0
    HIGH_NOISE_THRESHOLD = 850.0

    if laplacian_var > HIGH_NOISE_THRESHOLD or contrast < LOW_CONTRAST_THRESHOLD:
        selected_profile = OCR_PROFILES["noisy_or_low_contrast"]
    elif contrast > HIGH_CONTRAST_THRESHOLD:
        selected_profile = OCR_PROFILES["high_quality_digital"]
    else:
        selected_profile = OCR_PROFILES["standard_scan"]

    logging.info(f"Analysis: Contrast={contrast:.2f}, Noise={laplacian_var:.2f}. Selected Profile: '{selected_profile['description']}'")
    return selected_profile


# --- TEXT EXTRACTION ---
# No changes are needed in this function's logic, it already uses logging correctly.
def extract_text_from_pdf_with_ocr(pdf_path):
    # This function is correct as-is from your provided code.
    # ... (function content is the same)
    all_pages_data = []
    try:
        document = fitz.open(pdf_path)
        for page_num in range(document.page_count):
            try:
                page = document.load_page(page_num)
                page_data = {'page_num': page_num, 'text': '', 'is_ocr_page': False, 'ocr_word_details': []}
                text_from_page = page.get_text("text")

                has_significant_images = len(page.get_images()) > 0
                is_text_sparse = len(text_from_page.strip()) < 100

                if has_significant_images and is_text_sparse:
                    logging.info(f"Page {page_num + 1}: Image-based page detected. Attempting OCR...")
                    page_data['is_ocr_page'] = True
                    
                    temp_pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                    temp_img = Image.open(io.BytesIO(temp_pix.tobytes("png")))
                    open_cv_image_gray = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2GRAY)
                    profile = analyze_image_and_select_profile(open_cv_image_gray)
                    pix = page.get_pixmap(matrix=fitz.Matrix(profile['dpi'] / 72, profile['dpi'] / 72))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    final_gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                    if profile['denoise']:
                        final_gray_image = cv2.fastNlMeansDenoising(final_gray_image, None, h=10)
                    processed_image = cv2.adaptiveThreshold(final_gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, profile['thresh_block_size'], profile['thresh_c'])
                    ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DATAFRAME, config=f"--psm {profile['psm']}")
                    
                    ocr_data = ocr_data.dropna(subset=['text'])
                    ocr_data = ocr_data[ocr_data['text'].str.strip() != '']
                    page_ocr_text = ""
                    page_ocr_word_details = []
                    current_char_offset = 0
                    for _, row in ocr_data.iterrows():
                        word = str(row['text'])
                        x0, y0, w, h = row['left'], row['top'], row['width'], row['height']
                        scale_x, scale_y = page.rect.width / pix.width, page.rect.height / pix.height
                        word_rect = fitz.Rect(x0*scale_x, y0*scale_y, (x0+w)*scale_x, (y0+h)*scale_y)
                        page_ocr_word_details.append({'text': word, 'bbox': word_rect, 'start_char_in_ocr_text': current_char_offset, 'end_char_in_ocr_text': current_char_offset + len(word)})
                        page_ocr_text += word + " "
                        current_char_offset += len(word) + 1
                    page_data['text'] = page_ocr_text.strip()
                    page_data['ocr_word_details'] = page_ocr_word_details
                else:
                    logging.info(f"Page {page_num + 1}: Native text page detected.")
                    page_data['text'] = text_from_page
                 
                all_pages_data.append(page_data)

            except Exception as e:
                logging.error(f"Failed to process page {page_num + 1} in {pdf_path}. Error: {e}", exc_info=True)
                continue

        document.close()
        return all_pages_data
    except Exception as e:
        logging.error(f"Failed to open or read PDF {pdf_path}. Error: {e}", exc_info=True)
        return []

# --- ENTITY IDENTIFICATION (spaCy) ---
def identify_sensitive_entities(pages_data):
    if nlp is None:
        logging.error("spaCy model not loaded, cannot identify entities.")
        return []
    all_sensitive_entities = []
    PII_LABELS = ["PERSON", "NORP", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "MONEY", "TIME", "FAC"]
    for page_data in pages_data:
        doc = nlp(page_data['text'])
        for ent in doc.ents:
            if ent.label_ in PII_LABELS:
                all_sensitive_entities.append({"text": ent.text, "label": f"SPACY_{ent.label_}", "page_num": page_data['page_num'], "start_char": ent.start_char, "end_char": ent.end_char, "is_ocr_page": page_data['is_ocr_page'], "ocr_word_details": page_data.get('ocr_word_details', [])})
    return all_sensitive_entities

def identify_sensitive_entities_regex(text, page_num, is_ocr_page, ocr_word_details=None):
    """
    Identifies sensitive entities using a refined set of regular expressions
    with a focus on precision through capture groups.
    """
    regex_entities = []

    # --- HIGHLIGHT: Regex patterns are now designed with capture groups (...) ---
    # The goal is to capture ONLY the sensitive value, not the surrounding keywords.
    patterns = {
        # --- Contact Information ---
        "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE_NUMBER": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b",
        "POSTAL_ADDRESS": r"\b\d{1,5}\s(?:[A-Za-z]+\s){1,5}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|Place|Pl|Square|Sq|Terrace|Ter|Way|Wy|Circle|Cir)\b,?\s(?:[A-Z][a-z]+\s?){1,3},?\s[A-Z]{2}\s\d{5}(?:-\d{4})?\b",
        
        # --- National/Government IDs ---
        "AADHAAR_ID": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
        "PAN_ID": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        "SSN_US": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
        "PASSPORT_NUMBER": r"\b(?:[A-Z]{1}\d{7}|[A-Z]{2}\d{7}|[A-Z]{2}\d{6}[A-Z]{1})\b",
        "DRIVING_LICENSE_NUMBER": r"\b[A-Z0-9]{5,20}\b",
        "VOTER_ID_NUMBER": r"\b[A-Z]{3}\d{7}\b",
        "TAX_IDENTIFICATION_NUMBER": r"\b(?:EIN|TIN|ITIN|SSN)?:?\s?\d{2}[-.\s]?\d{7}\b|\b\d{9}\b|\b\d{3}-\d{2}-\d{4}\b",
                                     
        # --- Financial Data ---
        "BANK_ACCOUNT_NUMBER": r"\b\d{9,18}\b",
        "CREDIT_DEBIT_CARD": r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|6(?:011|5\d{2})\d{12}|3[47]\d{13}|3(?:0[0-5]|[68]\d)\d{11}|(?:2131|1800|35\d{3})\d{11})\b",
                             
        # --- Other Specific Identifiers ---
        "IP_ADDRESS_V4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "MAC_ADDRESS": r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b|\b(?:[0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}\b",
        "EMPLOYEE_ID": r"\bEID-\d{4,8}\b|\bEMP-\d{4,8}\b",
        "CUSTOMER_ID": r"\bCID-\d{4,8}\b|\bCUST-\d{4,8}\b",
        "VEHICLE_REG_NUMBER": r"\b[A-Z]{2}[0-9]{2}[A-Z]{2}\d{4}\b|\b[A-Z]{2}[0-9]{2}[A-Z]{1}\d{4}\b",
        
        # --- Dates (complementing spaCy's DATE where needed for specific formats) ---
        "DATE_SPECIFIC_DDMMYYYY": r"\b(?:0?[1-9]|[12][0-9]|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b",
        "DATE_SPECIFIC_MMDDYYYY": r"\b(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12][0-9]|3[01])[-/.](?:19|20)\d{2}\b",
        
        # --- Custom Keywords (like your 'PLL') ---
        "CUSTOM_KEYWORD_PLL": (r"\bPLL\b", re.IGNORECASE),
        
        # --- Other Attributes that might have specific text indicators ---
        "BLOOD_GROUP": r"\b(?:\(?\s*(A|B|AB|O)[+-]\s*\)?)\b",
        "DISABILITY_STATUS": r"\b(?:disabled|handicapped|impairment|special needs)\b",
        "SEXUAL_ORIENTATION": r"\b(?:heterosexual|homosexual|bisexual|asexual|pansexual)\b",
        "POLITICAL_BELIEFS": r"\b(?:democrat|republican|liberal|conservative|socialist|communist)\b",
        "RELIGIOUS_BELIEFS": r"\b(?:christian|muslim|hindu|buddhist|jewish|atheist)\b",
        "TRADE_UNION_MEMBERSHIP": r"\b(?:trade union member|unionized|union member)\b",
        "MARITAL_STATUS": r"\b(?:single|married|divorced|widowed)\b",
        
        # Generic identifiers for potentially sensitive terms that often appear with IDs
        "GENERIC_ID_TERM": r"\b(?:ID|No|Number|Ref|Reference|Code)\b",
        
        # Health Data keywords (highly contextual, very high false positive rate without NLP context)
        "HEALTH_KEYWORD": r"\b(?:medical|health|diagnosis|prescription|condition|illness|treatment|hospital|clinic)\b",

        "DOB_DDMMMYYYY": r"\b(?:0[1-9]|[12][0-9]|3[01])-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(?:19|20)\d{2}\b",
        "DOB_DDMMYYYY": r"\b(?:0[1-9]|[12][0-9]|3[01])/(?:0[1-9]|1[0-2])/(?:19|20)\d{2}\b",
        "GENDER": r"\b(?:\(?\s*(male|female|other|non-binary|transgender|trans|m|f)\s*\)?)\b",
        "AGE": r"\b(?:[Aa]ge[:\s]*|[A-Za-z]{0,5}\s*)?(?:\d{1,2})(?:\s?years?\s?old)?\b",
    }

    for label, pattern_item in patterns.items():
        pattern, flags = (pattern_item, 0) if isinstance(pattern_item, str) else pattern_item

        for match in re.finditer(pattern, text, flags):
            # --- HIGHLIGHT: The New Precision Logic ---
            # If the regex has capture groups, use the first one. Otherwise, use the whole match.
            # This ensures we only redact the value (e.g., "12345") not the label ("Employee ID: 12345")
            if match.re.groups > 0:
                start_char, end_char = match.span(1)
                matched_text = match.group(1)
            else:
                start_char, end_char = match.span(0)
                matched_text = match.group(0)

            # Final check to avoid redacting empty strings
            if not matched_text or not matched_text.strip():
                continue

            regex_entities.append({
                "text": matched_text,
                "label": f"REGEX_{label}",
                "page_num": page_num,
                "start_char": start_char,
                "end_char": end_char,
                "is_ocr_page": is_ocr_page,
                "ocr_word_details": ocr_word_details
            })
    return regex_entities

def redact_sensitive_info(pdf_path, detected_entities, output_pdf_path):
    try:
        document = fitz.open(pdf_path)
        total_redactions_applied = 0
        entities_by_page = {}
        for entity in detected_entities:
            page_num = entity['page_num']
            if page_num not in entities_by_page:
                entities_by_page[page_num] = []
            entities_by_page[page_num].append(entity)

        for page_num, page_entities in entities_by_page.items():
            page = document.load_page(page_num)
            
            # HIGHLIGHT: Using logging for consistent output
            logging.info(f"Applying redactions to page {page_num + 1}...")
            for entity in page_entities:
                if entity['is_ocr_page']:
                    ocr_word_details = entity.get('ocr_word_details')
                    if not ocr_word_details:
                        logging.warning(f"Skipping OCR entity '{entity['text']}' due to missing word details.")
                        continue
                    
                    matched_bboxes = []
                    for word in ocr_word_details:
                        if (entity['start_char'] < word['end_char_in_ocr_text']) and (entity['end_char'] > word['start_char_in_ocr_text']):
                            matched_bboxes.append(word['bbox'])
                    
                    if matched_bboxes:
                        for bbox in matched_bboxes:
                            if not bbox.is_empty:
                                page.add_redact_annot(bbox, text="█", fill=(0, 0, 0))
                                total_redactions_applied += 1
                    else:
                         logging.warning(f"Could not map OCR entity '{entity['text']}' to bounding boxes.")
                else: # Native text entity
                    text_instances = page.search_for(entity["text"])
                    for inst in text_instances:
                        if not inst.is_empty:
                            page.add_redact_annot(inst, text="█", fill=(0, 0, 0))
                            total_redactions_applied += 1

        if total_redactions_applied > 0:
            logging.info(f"Applying {total_redactions_applied} secure redaction marks across the document...")
            for page in document:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            document.save(output_pdf_path, garbage=4, deflate=True)
            logging.info(f"Successfully redacted document saved to: {output_pdf_path}")
            return True
        else:
            logging.info("No entities were found that could be mapped for redaction.")
            document.close()
            return False
    except Exception as e:
        logging.error(f"An error occurred during final redaction: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if nlp is None:
        sys.exit(1) # Exit if the spaCy model could not be loaded

    # HIGHLIGHT: Using logging for all runtime messages
    logging.info(f"Python Executable: {sys.executable}")
    logging.info(f"PyMuPDF (fitz) version: {fitz.__version__}")

    input_pdf_path = "Claude.pdf"
    output_pdf_path = "redacted4b.pdf"

    if not os.path.exists(input_pdf_path):
        logging.error(f"Error: Input PDF file not found at '{input_pdf_path}'.")
    else:
        logging.info(f"--- Starting New Redaction Process for: {input_pdf_path} ---")
        
        all_pages_data = extract_text_from_pdf_with_ocr(input_pdf_path)

        if all_pages_data:
            all_detected_entities = []
            logging.info("--- Identifying Sensitive Entities (spaCy + Regex) ---")
            for page_data in all_pages_data:
                all_detected_entities.extend(identify_sensitive_entities([page_data]))
                all_detected_entities.extend(identify_sensitive_entities_regex(
                    page_data['text'], page_data['page_num'], page_data['is_ocr_page'], page_data.get('ocr_word_details')
                ))
            
            unique_entities = list({(e['page_num'], e['start_char'], e['end_char']): e for e in all_detected_entities}.values())
            
            if unique_entities:
                logging.info(f"Detected {len(unique_entities)} unique sensitive entities. Applying redactions...")
                logging.info(f"\n--- TOTAL UNIQUE ENTITIES DETECTED: {len(unique_entities)} ---")
                for entity in sorted(unique_entities, key=lambda x: (x['page_num'], x['start_char'])):
                     print(f"  - Page {entity['page_num']+1} | Label: {entity['label']:<25} | Text: '{entity['text']}'")
                
                logging.info("\n--- Applying Redactions ---")
                redaction_success = redact_sensitive_info(input_pdf_path, unique_entities, output_pdf_path)
                if redaction_success:
                    logging.info("--- Redaction Process Completed Successfully ---")
                else:
                    logging.error("--- Redaction Process Failed ---")
            else:
                logging.info("No sensitive entities detected.")
        else:
            logging.error("Failed to extract any text from the PDF. Aborting.")