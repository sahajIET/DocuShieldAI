import fitz  # PyMuPDF is imported as 'fitz'
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

# --- IMPORTANT for Windows users ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Load spaCy Model ---
try:
    nlp = spacy.load("en_core_web_lg")
    print("spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please ensure you have run 'python -m spacy download en_core_web_lg'")
    nlp = None

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redaction_process.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- OCR PROFILES and ANALYSIS (No changes needed here) ---
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

# --- TEXT EXTRACTION (No changes needed here) ---
def extract_text_from_pdf_with_ocr(pdf_path):
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

# --- spaCy ENTITY IDENTIFICATION (No changes needed here) ---
def identify_sensitive_entities(pages_data):
    if nlp is None:
        logging.error("spaCy model not loaded, cannot identify entities.")
        return []
    all_sensitive_entities = []
    # Note: spaCy handles Full Name (PERSON), Nationality (NORP), Workplace (ORG), Location (GPE/LOC), DOB (DATE), etc.
    PII_LABELS = ["PERSON", "NORP", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "MONEY", "TIME", "FAC"]
    for page_data in pages_data:
        doc = nlp(page_data['text'])
        for ent in doc.ents:
            if ent.label_ in PII_LABELS:
                all_sensitive_entities.append({"text": ent.text, "label": ent.label_, "page_num": page_data['page_num'], "start_char": ent.start_char, "end_char": ent.end_char, "is_ocr_page": page_data['is_ocr_page'], "ocr_word_details": page_data.get('ocr_word_details', [])})
    return all_sensitive_entities

# --- STEP 9: EXPANDED REGEX FUNCTION ---
def identify_sensitive_entities_regex(text, page_num, is_ocr_page, ocr_word_details=None):
    """
    Identifies sensitive entities using a FLEXIBLE and ROBUST regex approach.
    It handles multiple keyword variations for key-value pairs and also finds
    uniquely formatted standalone PII.
    """
    regex_entities = []

    # This dictionary is now more comprehensive and flexible.
    # It uses non-capturing groups (?:...) to check for many possible labels.
    # The sensitive value itself is in a CAPTURE GROUP (...).
    patterns = {
        # --- Contact Information ---
        "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE_NUMBER": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b",
        # Address is very hard. This looks for "Address:" and captures the following multi-line text ending in a pincode.
        "POSTAL_ADDRESS": r"(?i)\b(?:Address|Location|Addr\.)[:\s-]+((?:[A-Za-z0-9\s,.-]+(?:\n|\s)){1,4}\b\d{5,7})",

        # --- National/Government IDs (High-Confidence) ---
        "AADHAAR_ID": r"\b(\d{4}[ -]?\d{4}[ -]?\d{4})\b",  # Unique pattern, can be standalone
        "PAN_ID": r"\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b",       # Unique pattern, can be standalone
        "SSN_US": r"(?i)\b(?:SSN|Social\sSecurity\sNo\.?)[:\s-]*(\d{3}[- ]?\d{2}[- ]?\d{4})\b",
        "PASSPORT_NUMBER": r"(?i)\b(?:Passport\sNo\.?)[:\s-]*([A-Z0-9]{6,15})\b",
        "DRIVING_LICENSE": r"(?i)\b(?:Driving\sLicense|DL\sNo\.?|License\sNumber)[\s:.-]*([A-Z0-9-]{8,20})\b",
        "VOTER_ID_NUMBER_IN": r"(?i)\b(?:Voter\sID|EPIC\sNo\.?)[:\s-]*([A-Z]{3}\d{7})\b",

        # --- Organizational & Financial IDs ---
        "EMPLOYEE_ID": r"(?i)\b(?:Employee\sID|Emp\sID|EID|Employee\sNo\.?)[:\s-]*([A-Z0-9-]{4,12})\b",
        "CUSTOMER_ID": r"(?i)\b(?:Customer\sID|Cust\sID|CID)[\s:.-]*([A-Z0-9-]{4,15})\b",
        "USER_ID": r"(?i)\b(?:User\s?ID|Username|Login)[\s:.-]*([a-zA-Z0-9_.-]{4,20})\b",
        "BANK_ACCOUNT_NUMBER": r"(?i)\b(?:Account\sNo|Acc\sNo|A/C)[\s:.-]*(\d{9,18})\b",
        "CREDIT_DEBIT_CARD": r"\b((?:4\d{3}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4})|(?:5[1-5]\d{2}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4})|(?:3[47]\d{2}[ -]?\d{6}[ -]?\d{5}))\b",

        # --- Health & Personal Data (Key-Value) ---
        # This requires a label to avoid redacting common words.
        "HEALTH_CONDITION": r"(?i)\b(?:Condition|Diagnosis|Illness)[\s:.-]+([A-Za-z\s,/.()-]+)(?=\n|\.|;)",
        "BLOOD_GROUP": r"(?i)\b(?:Blood\s(?:Type|Group)|B\.G\.)[:\s-]+((?:A|B|AB|O)\s?[+-](?:ve|sitive|gative)?)",
        
        # --- Other Identifiers ---
        "IP_ADDRESS": r"\b((?:[0-9]{1,3}\.){3}[0-9]{1,3})\b",
        "VEHICLE_REG_NUMBER": r"(?i)\b(?:Reg\sNo\.?|Vehicle\sNo\.?)[:\s-]*([A-Z]{2}[-.\s]?[0-9]{1,2}[-.\s]?[A-Z]{1,2}[-.\s]?[0-9]{1,4})\b",
        "FULL_NAME_WITH_TITLE": r"\b(?:Name\s*:\s*|Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+([A-Z][a-z]+\s(?:[A-Z][a-z]*\s)?[A-Z][a-z]+)",

    }

    for label, pattern in patterns.items():
        # The re.IGNORECASE flag is now handled directly in the regex strings with (?i) for simplicity
        for match in re.finditer(pattern, text):
            # --- CORE LOGIC (Same as before, now with better patterns) ---
            # If the regex has a capture group (most of them do), redact ONLY the group.
            # Otherwise, redact the entire match (for things like email).
            if match.lastindex and match.lastindex > 0:
                # Group(s) exist. We prioritize the first group as the sensitive value.
                start_char, end_char = match.span(1)
                matched_text = match.group(1)
            else:
                # No capture groups, the entire pattern is sensitive.
                start_char, end_char = match.span(0)
                matched_text = match.group(0)

            # Final check to avoid redacting empty strings or just whitespace
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

# --- [Keep all your other code: imports, other functions, main block, etc.] ---
# --- vvv REPLACE THIS ENTIRE FUNCTION vvv ---

def redact_sensitive_info(pdf_path, detected_entities, output_pdf_path):
    """
    Applies SECURE redactions to the PDF. It not only draws a black box
    but also scrubs the underlying text, replacing it with a block character
    to prevent copy-pasting of sensitive data.
    """
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
            logging.info(f"--- Processing Page {page_num + 1} for secure redaction ---")
            for entity in page_entities:
                entity_text = entity["text"]
                is_ocr_entity = entity['is_ocr_page']

                if is_ocr_entity:
                    ocr_word_details = entity.get('ocr_word_details')
                    if not ocr_word_details:
                        logging.warning(f"    - Skipping OCR entity '{entity_text}' due to missing word details.")
                        continue
                    
                    matched_bboxes = []
                    for word in ocr_word_details:
                        if (entity['start_char'] < word['end_char_in_ocr_text']) and (entity['end_char'] > word['start_char_in_ocr_text']):
                            matched_bboxes.append(word['bbox'])
                    
                    if matched_bboxes:
                        logging.info(f"    - Redacting OCR entity '{entity['label']}: {entity_text}' with {len(matched_bboxes)} boxes.")
                        for bbox in matched_bboxes:
                            margin = 1
                            precise_rect = fitz.Rect(bbox.x0 - margin, bbox.y0 - margin, bbox.x1 + margin, bbox.y1 + margin).intersect(page.rect)
                            if not precise_rect.is_empty:
                                # *** SECURITY FIX APPLIED HERE ***
                                page.add_redact_annot(precise_rect, text="█", fill=(0, 0, 0))
                                total_redactions_applied += 1
                    else:
                         logging.warning(f"    - Could not map OCR entity '{entity_text}' to bounding boxes.")

                else: # Native text entity
                    logging.info(f"    - Redacting native text entity '{entity['label']}: {entity_text}'.")
                    text_instances = page.search_for(entity_text, quads=False) # Use rectangles for simplicity
                    for inst in text_instances:
                        margin = 2 
                        expanded_rect = fitz.Rect(inst.x0 - margin, inst.y0 - margin, inst.x1 + margin, inst.y1 + margin).intersect(page.rect)
                        if not expanded_rect.is_empty:
                            # *** SECURITY FIX APPLIED HERE ***
                            page.add_redact_annot(expanded_rect, text="█", fill=(0, 0, 0))
                            total_redactions_applied += 1

        if total_redactions_applied > 0:
            logging.info(f"\nApplying {total_redactions_applied} secure redaction marks across the document...")
            # Apply the redactions, which now includes scrubbing the text
            for page in document:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

            # Save the document with garbage collection to remove old, unreferenced data
            document.save(output_pdf_path, garbage=4, deflate=True) 
            logging.info(f"Securely redacted PDF saved to: {output_pdf_path}")
            return True
        else:
            logging.info("\nNo sensitive entities were found or could be mapped for redaction.")
            document.close()
            return False
    except Exception as e:
        logging.error(f"An error occurred during the final redaction and saving process: {e}", exc_info=True)
        return False

# --- ^^^ REPLACE THE FUNCTION ABOVE ^^^ ---
# --- [Keep all your other code: main block, etc.] ---

# --- MAIN EXECUTION BLOCK (No changes needed here) ---
if __name__ == "__main__":
    logging.info(f"Python Executable: {sys.executable}")
    logging.info(f"PyMuPDF (fitz) version: {fitz.__version__}")

    # --- CONFIGURATION ---
    input_pdf_path = "GDPR_PII_Mixed_Document.pdf"  # Replace with your input PDF
    output_pdf_path = "redacted_document_comprehensive.pdf" # New output file name

    if not os.path.exists(input_pdf_path):
        logging.error(f"Error: Input PDF file not found at '{input_pdf_path}'.")
    else:
        logging.info(f"\n--- Starting Comprehensive PII Processing for: {input_pdf_path} ---")

        # 1. Extract text using the auto-tuning OCR pipeline
        all_pages_data = extract_text_from_pdf_with_ocr(input_pdf_path)

        if all_pages_data:
            all_detected_entities = []

            logging.info("\n--- Identifying Sensitive Entities (Phase 1: spaCy NER) ---")
            # The existing spaCy function is called for each page implicitly in the loop
            
            logging.info("\n--- Identifying Sensitive Entities (Phase 2: Comprehensive Regex) ---")
            for page_data in all_pages_data:
                # Get entities from spaCy (Names, Places, Dates, etc.)
                spacy_entities = identify_sensitive_entities([page_data])
                all_detected_entities.extend(spacy_entities)
                
                # Get entities from our new comprehensive Regex function
                regex_entities = identify_sensitive_entities_regex(
                    page_data['text'], page_data['page_num'], page_data['is_ocr_page'], page_data.get('ocr_word_details')
                )
                all_detected_entities.extend(regex_entities)
            
            # Remove duplicate entities that might be found by both methods or multiple times
            unique_entities = list({
                (e['text'], e['label'], e['page_num'], e['start_char'], e['end_char']): e 
                for e in all_detected_entities
            }.values())
            
            if unique_entities:
                logging.info(f"\n--- TOTAL UNIQUE ENTITIES DETECTED: {len(unique_entities)} ---")
                for entity in sorted(unique_entities, key=lambda x: (x['page_num'], x['start_char'])):
                     print(f"  - Page {entity['page_num']+1} | Label: {entity['label']:<25} | Text: '{entity['text']}'")
                
                logging.info("\n--- Applying Redactions ---")
                redaction_success = redact_sensitive_info(input_pdf_path, unique_entities, output_pdf_path)

                if redaction_success:
                    logging.info("\nRedaction process completed successfully!")
                else:
                    logging.info("\nRedaction process failed during file saving.")
            else:
                logging.info("No sensitive entities detected across all pages.")
        else:
            logging.error("Failed to extract any text from the PDF. Halting process.")