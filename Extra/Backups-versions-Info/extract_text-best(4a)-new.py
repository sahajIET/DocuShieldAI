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

# --- IMPORTANT for Windows users: ---
# If tesseract.exe is not in your PATH, you might need to specify its path.
# For example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Uncomment the line below and replace with your actual path if needed.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Load the spaCy English language model ---
try:
    nlp = spacy.load("en_core_web_lg")
    print("spaCy model 'en_core_web_lg' loaded successfully.") # Changed from print
except Exception as e:
    print(f"Error loading spaCy model: {e}") # Changed from print
    print("Please ensure you have run 'python -m spacy download en_core_web_lg'") # Changed from print
    nlp = None

# --- 1. SETUP THE LOGGER (No changes needed, already well-configured) ---
# Configure the logger to write to a file and the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redaction_process.log"), # Log to a file
        logging.StreamHandler(sys.stdout)             # Log to the console
    ]
)

# --- OCR_PROFILES and analyze_image_and_select_profile function (No changes) ---
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

# --- 2. UPDATE FUNCTIONS TO USE THE LOGGER ---

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
                    
                    # (The auto-tuning logic is the same)
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

                    # (The text reconstruction logic is the same)
                    ocr_data = ocr_data.dropna(subset=['text'])
                    # Ensure 'text' column is string type before stripping
                    ocr_data['text'] = ocr_data['text'].astype(str) # Added/Confirmed
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
                continue # Continue to the next page

        document.close()
        return all_pages_data
    except Exception as e:
        logging.error(f"Failed to open or read PDF {pdf_path}. Error: {e}", exc_info=True)
        return []

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
                all_sensitive_entities.append({"text": ent.text, "label": f"SPACY_{ent.label_}", "page_num": page_data['page_num'], "start_char": ent.start_char, "end_char": ent.end_char, "is_ocr_page": page_data['is_ocr_page'], "ocr_word_details": page_data.get('ocr_word_details', [])}) # Modified label prefix
    return all_sensitive_entities


def identify_sensitive_entities_regex(text, page_num, is_ocr_page, ocr_word_details=None):
    """
    Identifies sensitive entities using a refined set of regular expressions.
    """
    regex_entities = []

    patterns = {
    "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE_NUMBER": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b",
    "POSTAL_ADDRESS": r"\b\d{1,5}\s(?:[A-Za-z]+\s){1,5}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|Place|Pl|Square|Sq|Terrace|Ter|Way|Wy|Circle|Cir)\b,?\s(?:[A-Z][a-z]+\s?){1,3},?\s[A-Z]{2}\s\d{5}(?:-\d{4})?\b",
    "AADHAAR_ID": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
    "PAN_ID": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
    "SSN_US": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
    "PASSPORT_NUMBER": r"\b(?:[A-Z]{1}\d{7}|[A-Z]{2}\d{7}|[A-Z]{2}\d{6}[A-Z]{1})\b",
    "DRIVING_LICENSE": r"(?i)\b(?:Driving\sLicense|DL\sNo\.?|License\sNumber)[\s:.-]*([A-Z0-9-]{8,20})\b",
    "VOTER_ID_NUMBER_IN": r"(?i)\b(?:Voter\sID|EPIC\sNo\.?)[:\s-]*([A-Z]{3}\d{7})\b",
    "TAX_IDENTIFICATION_NUMBER": r"\b(?:EIN|TIN|ITIN|SSN)?:?\s?\d{2}[-.\s]?\d{7}\b|\b\d{9}\b|\b\d{3}-\d{2}-\d{4}\b",
    "BANK_ACCOUNT_NUMBER": r"(?i)\b(?:Account\sNo|Acc\sNo|A/C)[\s:.-]*(\d{9,18})\b",
    "CREDIT_DEBIT_CARD": r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|6(?:011|5\d{2})\d{12}|3[47]\d{13}|3(?:0[0-5]|[68]\d)\d{11}|(?:2131|1800|35\d{3})\d{11})\b",
    "IP_ADDRESS_V4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "MAC_ADDRESS": r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b|\b(?:[0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}\b",
    "EMPLOYEE_ID": r"\bEID-\d{4,8}\b|\bEMP-\d{4,8}\b",
    "CUSTOMER_ID": r"\bCID-\d{4,8}\b|\bCUST-\d{4,8}\b",
    # "BLOOD_GROUP": r"\b(?:A|B|AB|O)[+-]\b", # Removed the optional parentheses and spaces for stricter match
    "BLOOD_GROUP": r"(?i)\b(?:Blood\s(?:Type|Group)|B\.G\.)[:\s-]+((?:A|B|AB|O)\s?[+-](?:ve|sitive|gative)?)",
    "DISABILITY_STATUS": r"\b(?:disabled|handicapped|impairment|special\sneeds)\b", # Added \s for special needs
    "SEXUAL_ORIENTATION": r"\b(?:heterosexual|homosexual|bisexual|asexual|pansexual)\b",
    "POLITICAL_BELIEFS": r"\b(?:democrat|republican|liberal|conservative|socialist|communist)\b",
    "RELIGIOUS_BELIEFS": r"\b(?:christian|muslim|hindu|buddhist|jewish|atheist)\b",
    # "TRADE_UNION_MEMBERSHIP": r"\b(?:trade\sunion\smember|unionized|union\smember)\b", # Added \s for multi-word
    "MARITAL_STATUS": r"\b(?:single|married|divorced|widowed)\b",
    # "GENERIC_ID_TERM": r"\b(?:ID|No|Number|Ref|Reference|Code)\b",
    #"GENERIC_ID_TERM": r"\b(?:ID|No|Number|Ref|Reference|Code)\b",
    #"GENERIC_ID_TERM": (r"\b(?:ID|No|Number|Ref|Reference|Code)\b", re.IGNORECASE),
    "HEALTH_KEYWORD": r"\b(?:medical|health|diagnosis|prescription|condition|illness|treatment|hospital|clinic)\b",
    "DOB_DDMMMYYYY": r"\b(?:0[1-9]|[12][0-9]|3[01])-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(?:19|20)\d{2}\b",
    "DOB_DDMMYYYY": r"\b(?:0[1-9]|[12][0-9]|3[01])/(?:0[1-9]|1[0-2])/(?:19|20)\d{2}\b",
    "GENDER": r"\b(?:male|female|other|non-binary|transgender|trans)\b", # Removed 'm' and 'f' due to high false positive rate
    "AGE": r"\b(?:age\s*|yrs?\s*|y\.o\s*)?(\d{1,2})\b", # Stricter AGE, removes the problematic [A-Za-z]{0,5}
    #"NUMBER": "\d+(\.\d+)?",
    }

    for label, pattern_item in patterns.items():
        # Unpack pattern and potential flags
        if isinstance(pattern_item, tuple):
            pattern, flags = pattern_item
        else:
            pattern, flags = pattern_item, 0

        # compiled_pattern = re.compile(pattern, flags) # Compile once for efficiency and to see the regex object
        # for match in compiled_pattern.finditer(text):
        #     matched_string = match.group(0)
        #     start_char = match.start()
        #     end_char = match.end()

        for match in re.finditer(pattern, text, flags):
            regex_entities.append({
                "text": match.group(0),
                "label": f"REGEX_{label}", # Modified label prefix
                "page_num": page_num,
                "start_char": match.start(),
                "end_char": match.end(),
                "is_ocr_page": is_ocr_page,
                "ocr_word_details": ocr_word_details
            })
        
            # if "id" in matched_string.lower() or "ID" in matched_string:
            #     logging.debug(f"DEBUG MATCH: Pattern '{label}' (regex: '{pattern}', flags: {flags}) matched '{matched_string}' at char {start_char}-{end_char} on page {page_num+1}.")
            #     # Print a wider context to see surrounding characters
            #     context_start = max(0, start_char - 20)
            #     context_end = min(len(text), end_char + 20)
            #     logging.debug(f"DEBUG CONTEXT: Text around match: '...{text[context_start:start_char]}***{text[start_char:end_char]}***{text[end_char:context_end]}...'")

            # regex_entities.append({
            #     "text": matched_string,
            #     "label": f"REGEX_{label}",
            #     "page_num": page_num,
            #     "start_char": start_char,
            #     "end_char": end_char,
            #     "is_ocr_page": is_ocr_page,
            #     "ocr_word_details": ocr_word_details
            # })
    return regex_entities


def redact_sensitive_info(pdf_path, detected_entities, output_pdf_path):
    """
    Redacts sensitive information by applying precise, individual redactions for each word,
    resulting in cleaner, non-vague redaction boxes.

    Args:
        pdf_path (str): Path to the input PDF file.
        detected_entities (list): List of detected entity dictionaries.
        output_pdf_path (str): Path to save the redacted PDF file.

    Returns:
        bool: True if redaction was successful, False otherwise.
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
            
            logging.info(f"--- Processing Page {page_num + 1} for secure Redaction ---") # Changed from print
            for entity in page_entities:
                entity_text = entity["text"]
                is_ocr_entity = entity['is_ocr_page']

                if is_ocr_entity:
                    # --- NEW PRECISION LOGIC FOR OCR ---
                    ocr_word_details = entity.get('ocr_word_details')
                    if not ocr_word_details:
                        logging.warning(f"    - Warning: Skipping OCR entity '{entity_text}' on page {page_num + 1} due to missing word details.") # Changed from print
                        continue
                    
                    matched_bboxes = []
                    for word in ocr_word_details:
                        # Check if the character span of the OCR word overlaps with the entity's span
                        if (entity['start_char'] < word['end_char_in_ocr_text']) and \
                           (entity['end_char'] > word['start_char_in_ocr_text']):
                            matched_bboxes.append(word['bbox'])
                    
                    if matched_bboxes:
                        logging.info(f"    - Redacting OCR entity '{entity_text}' (Label: {entity['label']}) with {len(matched_bboxes)} individual boxes on page {page_num + 1}.") # Changed from print
                        # **APPLY REDACTION FOR EACH MATCHED WORD SEPARATELY**
                        for bbox in matched_bboxes:
                            # You can still add a small margin for safety
                            margin = 1
                            precise_rect = fitz.Rect(
                                bbox.x0 - margin, bbox.y0 - margin,
                                bbox.x1 + margin, bbox.y1 + margin
                            ).intersect(page.rect) # Ensure it's within page bounds
                            
                            if not precise_rect.is_empty: # Only add non-empty rects
                                page.add_redact_annot(precise_rect,text="█", fill=(0, 0, 0))
                                total_redactions_applied += 1
                    else:
                         logging.warning(f"    - Warning: Could not map OCR entity '{entity_text}' (Label: {entity['label']}) to bounding boxes on page {page_num + 1}.") # Changed from print

                else: # This is a native text entity
                    logging.info(f"    - Redacting native text entity '{entity_text}' (Label: {entity['label']}) on page {page_num + 1}.") # Changed from print
                    text_instances = page.search_for(entity_text) # was changed here
                    for inst in text_instances:
                        margin = 2 
                        expanded_rect = fitz.Rect(
                            inst.x0 - margin, inst.y0 - margin,
                            inst.x1 + margin, inst.y1 + margin
                        ).intersect(page.rect)
                        
                        if not expanded_rect.is_empty:
                            page.add_redact_annot(expanded_rect,text="█", fill=(0, 0, 0))
                            total_redactions_applied += 1

        if total_redactions_applied > 0:
            logging.info(f"\nApplying {total_redactions_applied} secure redaction marks across the document...") # Changed from print
            for page in document:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                # this will replace the pixel with the 'text' block images=fitz.PDF_REDACT_IMAGE_NONE , text="█"
                # this will just deleted the pixels able to copy other details images=fitz.PDF_REDACT_IMAGE_PIXELS
                # adding text="█" will make the things turn into unrecognized text like '?'
            document.save(output_pdf_path, garbage=4, deflate=True) 
            logging.info(f"Successfully applied redactions.") # Changed from print
            logging.info(f"Redacted PDF saved to: {output_pdf_path}") # Changed from print
            return True
        else:
            logging.info("\nNo sensitive entities were found or could be mapped to locations for redaction.") # Changed from print
            document.close()
            return False

    except Exception as e:
        logging.error(f"An error occurred during redaction: {e}", exc_info=True) # Changed from print
        return False


if __name__ == "__main__":
    logging.info(f"Python Executable: {sys.executable}") # Changed from print
    try:
        logging.info(f"PyMuPDF (fitz) version: {fitz.__version__}") # Changed from print
    except ImportError:
        logging.warning("PyMuPDF (fitz) is not imported or installed correctly.") # Changed from print

    input_pdf_path = "10th.pdf"
    output_pdf_path = "redacted4a.pdf"

    if not os.path.exists(input_pdf_path):
        logging.error(f"Error: Input PDF file not found at '{input_pdf_path}'.") # Changed from print
    else:
        logging.info(f"\n--- Starting Auto-Tuned Processing for: {input_pdf_path} ---") # Changed from print

        # Step 1: Extract text using the new auto-tuning function
        all_pages_data = extract_text_from_pdf_with_ocr(input_pdf_path)

        if all_pages_data:
            all_detected_entities = []

            logging.info("\n--- Identifying Sensitive Entities (spaCy & Regex) ---") # Changed from print
            for page_data in all_pages_data:
                spacy_entities = identify_sensitive_entities([page_data])
                all_detected_entities.extend(spacy_entities)
                regex_entities = identify_sensitive_entities_regex(
                    page_data['text'], page_data['page_num'], page_data['is_ocr_page'], page_data.get('ocr_word_details')
                )
                all_detected_entities.extend(regex_entities)
            
            # Use a tuple of immutable fields for uniqueness
            unique_entities = {
                (e['text'], e['label'], e['page_num'], e['start_char'], e['end_char']): e 
                for e in all_detected_entities
            }.values()
            
            if unique_entities:
                logging.info(f"Detected {len(unique_entities)} unique sensitive entities.") # Changed from print
                # Log detailed unique entities
                # for entity in unique_entities:
                #     logging.debug(f"  - Label: {entity['label']:<30} | Text: '{entity['text']}' | Page: {entity['page_num'] + 1}") # Changed to debug for detail

                for entity in sorted(unique_entities, key=lambda x: (x['page_num'], x['start_char'])):
                     print(f"  - Page {entity['page_num']+1} | Label: {entity['label']:<25} | Text: '{entity['text']}'")
                logging.info("\n--- Applying Redaction ---") # Changed from print
                redaction_success = redact_sensitive_info(input_pdf_path, list(unique_entities), output_pdf_path)

                if redaction_success:
                    logging.info("\nRedaction process completed successfully!") # Changed from print
                else:
                    logging.error("\nRedaction process failed.") # Changed from print
            else:
                logging.info("No sensitive entities detected.") # Changed from print
        else:
            logging.error("Failed to extract text from the PDF.") # Changed from print