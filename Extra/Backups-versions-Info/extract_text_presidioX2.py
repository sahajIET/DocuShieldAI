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
# ... after your other imports
from presidio_analyzer import AnalyzerEngine
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

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detects the skew angle of the text in an image and corrects it.
    This is a crucial step for improving OCR accuracy on scanned documents.
    """
    # --- HIGHLIGHT: Deskewing Logic ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray) # Invert colors so text is white on black background
    
    # Get the coordinates of all non-zero (text) pixels
    coords = np.column_stack(np.where(gray > 0))
    
    # Get the minimum area bounding rectangle of the text block
    angle = cv2.minAreaRect(coords)[-1]

    # The `minAreaRect` angle can be [-90, 0). We need to adjust it.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # If the skew is negligible, don't bother rotating
    if abs(angle) < 0.1:
        logging.info("Deskewing not needed, angle is negligible.")
        return image

    logging.info(f"Detected skew angle: {angle:.2f} degrees. Correcting...")
    
    # Rotate the original image to correct the skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
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
                    
                    # --- HIGHLIGHT: CORRECTED LOGICAL ORDER ---

                    # STEP 1: Render the page and create the base OpenCV image FIRST.
                    # We use a standard DPI here for analysis. The profile's DPI is less critical
                    # than its preprocessing settings.
                    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    
                    # STEP 2: Now that open_cv_image exists, create the grayscale version for analysis.
                    open_cv_image_gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                    
                    # STEP 3: Analyze the grayscale image to select the best processing profile.
                    profile = analyze_image_and_select_profile(open_cv_image_gray)

                    # STEP 4: Apply deskewing to the original color image.
                    deskewed_image = (open_cv_image)
                    
                    # STEP 5: Continue preprocessing on the deskewed image.
                    final_gray_image = cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2GRAY)
                    if profile['denoise']:
                        final_gray_image = cv2.fastNlMeansDenoising(final_gray_image, None, h=10)
                    processed_image = cv2.adaptiveThreshold(final_gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, profile['thresh_block_size'], profile['thresh_c'])

                    # STEP 6: Run Tesseract with the advanced configuration.
                    tesseract_config = (
                        f"--psm {profile['psm']} "
                        f"-c load_system_dawg=0 "
                        f"-c load_freq_dawg=0"
                    )
                    ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DATAFRAME, config=tesseract_config)

                    # (The rest of the text reconstruction logic remains the same)
                    ocr_data = ocr_data.dropna(subset=['text'])
                    ocr_data['text'] = ocr_data['text'].astype(str)
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
    Identifies sensitive entities using a *refined* set of regular expressions.
    We rely on Presidio for broad patterns and use this for very specific cases.
    """
    regex_entities = []

    # --- MODIFIED PATTERNS ---
    patterns = {
        # Keep high-confidence patterns
        "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PAN_ID": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        "AADHAAR_ID": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
        "IP_ADDRESS_V4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "MAC_ADDRESS": r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b|\b(?:[0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}\b",
        
        # Make AGE pattern stricter (keyword is no longer optional)
        "AGE": r"\b(?:age|years\sold|yrs|y\.o\.)[\s:]*(\d{1,2})\b",
        
        # Keep special keyword-based patterns
        "BLOOD_GROUP": r"(?i)(?:\b(?:blood\s(?:group|type)|b\.g\.)[:\s-]*)?[\(\[]?\s*(A|B|AB|O)\s?[+-](?:ve|sitive|gative)?\s*[\)\]]?",
        "ALPHANUMERIC_ID_GENERIC": r"\b(?=[a-zA-Z0-9-]{6,25}\b)(?:[a-zA-Z]+[-_]*\d+|\d+[-_]*[a-zA-Z]+)[a-zA-Z0-9-]*\b",
        "GENDER": r"(?i)\b(?:male|female)\b",

        # REMOVED: "NUMBER", "PHONE_NUMBER", "BANK_ACCOUNT_NUMBER", etc.
        # We will let Presidio handle these more intelligently.
    }

    for label, pattern_item in patterns.items():
        # ... (the rest of your function remains the same) ...
        if isinstance(pattern_item, tuple):
            pattern, flags = pattern_item
        else:
            pattern, flags = pattern_item, 0
            
        for match in re.finditer(pattern, text, flags):
            regex_entities.append({
                "text": match.group(0),
                "label": f"REGEX_{label}",
                "page_num": page_num,
                "start_char": match.start(),
                "end_char": match.end(),
                "is_ocr_page": is_ocr_page,
                "ocr_word_details": ocr_word_details
            })
    return regex_entities

# ==============================================================================
# --- STEP 4: ADD PRESIDIO AND THE VALIDATION LAYER FUNCTIONS ---
# ==============================================================================

# --- A. Allow/Deny lists for post-processing ---
# Add common words that spaCy incorrectly flags as PERSON or ORG
DENY_LIST_TERMS = {
    "inc", "ltd", "llc", "corp", "corporation", "gmbh", "pvt", "ltd", # Company suffixes
    "fig", "figure", "table", "appendix", "chapter", "section",    # Document structure
    "note", "notes", "summary", "introduction", "conclusion", "abstract",
    # Add any other specific false positives you frequently encounter in your documents
    "gemini", "pytesseract", "spacy", "tesseract"
}

def identify_entities_with_presidio(analyzer, text, page_num, is_ocr_page, ocr_word_details=None):
    """
    Uses Microsoft Presidio to identify a wide range of PII with higher confidence.
    """
    presidio_entities = []
    try:
        # Analyze the text for all entities Presidio knows about
        results = analyzer.analyze(text=text, language='en')
        
        for res in results:
            presidio_entities.append({
                "text": text[res.start:res.end],
                "label": f"PRESIDIO_{res.entity_type}",
                "page_num": page_num,
                "start_char": res.start,
                "end_char": res.end,
                "score": res.score, # Presidio provides a confidence score!
                "is_ocr_page": is_ocr_page,
                "ocr_word_details": ocr_word_details
            })
    except Exception as e:
        logging.error(f"Presidio analysis failed on page {page_num + 1}: {e}")
    return presidio_entities

def is_context_relevant(text, entity_start, entity_end, window_size=30):
    """
    Checks if the context around an entity suggests it is sensitive.
    """
    context_start = max(0, entity_start - window_size)
    context_end = min(len(text), entity_end + window_size)
    context_window = text[context_start:context_end].lower()

    # Keywords that CONFIRM sensitivity for an ambiguous term (like a number or name)
    confirm_keywords = [
        'account', 'acct', 'a/c', 'card', 'ssn', 'id', 'license', 'passport',
        'member', 'employee', 'emp', 'customer', 'ref', 'invoice', 'po #', 'p.o.',
        'phone', 'tel', 'mobile', 'fax', 'email', 'name', 'mr.', 'mrs.', 'ms.'
    ]

    # Keywords that suggest it's just regular data
    deny_keywords = [
        'page', 'chapter', 'section', 'fig', 'figure', 'table', 'quantity',
        'qty', 'item', 'step', 'version', 'v.', 'rev', 'line', 'row', 'model'
    ]

    if any(keyword in context_window for keyword in deny_keywords):
        return False  # Likely a false positive

    if any(keyword in context_window for keyword in confirm_keywords):
        return True # Likely a true positive

    return False # If no keywords found, remain neutral/negative

def get_semantic_pii_score(text, entity_start, entity_end, nlp_model):
    """
    Calculates a score based on semantic similarity of context words to PII concepts.
    """
    pii_profile = nlp_model("personal private identity financial health contact address security account")
    
    context_start = max(0, entity_start - 10)
    context_end = min(len(text), entity_end + 10)
    context_text = text[context_start:entity_start] + " " + text[entity_end:context_end]

    if not context_text.strip(): return 0.0
    context_doc = nlp_model(context_text)
    
    context_doc_no_stopwords = [token for token in context_doc if not token.is_stop and not token.is_punct]
    if not context_doc_no_stopwords: return 0.0
    
    final_doc = nlp_model(' '.join([token.text for token in context_doc_no_stopwords]))
    if not final_doc.vector_norm: return 0.0
    
    return final_doc.similarity(pii_profile)

def post_process_and_validate_entities(entities, full_text_by_page, nlp_model):
    """
    The main validation filter. It takes all detected entities and removes likely false positives.
    """
    validated_entities = []
    processed_spans = set()

    # Sort by length descending to prioritize larger, more specific entities
    for entity in sorted(entities, key=lambda x: len(x['text']), reverse=True):
        entity_span = (entity['page_num'], entity['start_char'], entity['end_char'])
        if any((s[0] == entity_span[0] and s[1] < entity_span[2] and s[2] > entity_span[1]) for s in processed_spans):
            continue
        
        text = entity['text'].lower().strip()
        label = entity['label']
        page_num = entity['page_num']
        full_text = full_text_by_page[page_num]

        # --- RULE 1: Hard Deny List ---
        if text in DENY_LIST_TERMS:
            logging.info(f"FILTERED (Deny List): '{entity['text']}' ({label})")
            continue

        # --- RULE 2: Score-based validation for Presidio entities ---
        if 'PRESIDIO' in label and entity.get('score', 0) < 0.0:
             logging.info(f"FILTERED (Low Presidio Score: {entity['score']:.2f}): '{entity['text']}' ({label})")
             continue

        # --- RULE 3: Contextual Validation for Ambiguous Entities (like numbers, generic names) ---
        ambiguous_labels = {"SPACY_CARDINAL", "SPACY_PERSON", "SPACY_ORG"}
        if label in ambiguous_labels:
            if is_context_relevant(full_text, entity['start_char'], entity['end_char']):
                logging.info(f"VALIDATED (Context Keyword): '{entity['text']}' ({label})")
            else:
                semantic_score = get_semantic_pii_score(full_text, entity['start_char'], entity['end_char'], nlp_model)
                if semantic_score > 0.60: # Tunable threshold
                    logging.info(f"VALIDATED (Semantic Score: {semantic_score:.2f}): '{entity['text']}' ({label})")
                else:
                    logging.info(f"FILTERED (Low Context/Semantics: {semantic_score:.2f}): '{entity['text']}' ({label})")
                    continue # Filter out this entity
        
        validated_entities.append(entity)
        processed_spans.add(entity_span)

    return validated_entities

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
                                page.add_redact_annot(precise_rect, fill=(0, 0, 0))
                                total_redactions_applied += 1
                    else:
                         logging.warning(f"    - Warning: Could not map OCR entity '{entity_text}' (Label: {entity['label']}) to bounding boxes on page {page_num + 1}.") # Changed from print

                else: # This is a native text entity
                    logging.info(f"    - Redacting native text entity '{entity_text}' (Label: {entity['label']}) on page {page_num + 1}.") # Changed from print
                    text_instances = page.search_for(entity_text, quads=True) # Use quads=True
                    for quad in text_instances:
                        page.add_redact_annot(quad, fill=(0, 0, 0))
                        total_redactions_applied += 1
                    # text_instances = page.search_for(entity_text) # was changed here
                    # for inst in text_instances:
                    #     margin = 2 
                    #     expanded_rect = fitz.Rect(
                    #         inst.x0 - margin, inst.y0 - margin,
                    #         inst.x1 + margin, inst.y1 + margin
                    #     ).intersect(page.rect)
                        
                    #     if not expanded_rect.is_empty:
                    #         page.add_redact_annot(expanded_rect, fill=(0, 0, 0))
                    #         total_redactions_applied += 1

        if total_redactions_applied > 0:
            logging.info(f"\nApplying {total_redactions_applied} secure redaction marks across the document...") # Changed from print
            for page in document:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
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
    logging.info(f"Python Executable: {sys.executable}")
    try:
        logging.info(f"PyMuPDF (fitz) version: {fitz.__version__}")
    except ImportError:
        logging.warning("PyMuPDF (fitz) is not imported or installed correctly.")

    input_pdf_path = "10th.pdf" # Make sure to use your input file
    output_pdf_path = "redacted_finalX2.pdf"

    if not os.path.exists(input_pdf_path):
        logging.error(f"Error: Input PDF file not found at '{input_pdf_path}'.")
    elif nlp is None:
        logging.error("spaCy model 'en_core_web_lg' not loaded. Aborting process.")
    else:
        # --- 1. SETUP PRESIDIO ANALYZER ---
        analyzer = AnalyzerEngine()
        
        logging.info(f"\n--- Starting Advanced Processing for: {input_pdf_path} ---")
        
        # --- 2. EXTRACT TEXT ---
        all_pages_data = extract_text_from_pdf_with_ocr(input_pdf_path)
        
        if all_pages_data:
            raw_detected_entities = []
            full_text_by_page = {p['page_num']: p['text'] for p in all_pages_data}

            # --- 3. DETECT POTENTIAL ENTITIES (Presidio + spaCy + Regex) ---
            logging.info("\n--- Phase 1: Detecting All Potential Entities ---")
            for page_data in all_pages_data:
                # Use Presidio for robust detection
                presidio_entities = identify_entities_with_presidio(analyzer, page_data['text'], page_data['page_num'], page_data['is_ocr_page'], page_data.get('ocr_word_details'))
                raw_detected_entities.extend(presidio_entities)

                # Use spaCy for general NER
                #spacy_entities = identify_sensitive_entities([page_data])
                #raw_detected_entities.extend(spacy_entities)

                # Use our refined regex for highly specific patterns
                regex_entities = identify_sensitive_entities_regex(page_data['text'], page_data['page_num'], page_data['is_ocr_page'], page_data.get('ocr_word_details'))
                raw_detected_entities.extend(regex_entities)

            # --- 4. VALIDATE AND FILTER ENTITIES ---
            logging.info(f"\n--- Phase 2: Validating {len(raw_detected_entities)} Potential Entities to Remove False Positives ---")
            validated_entities = post_process_and_validate_entities(raw_detected_entities, full_text_by_page, nlp)

            # --- 5. REDACT THE FINAL, VALIDATED ENTITIES ---
            if validated_entities:
                logging.info(f"\n--- Phase 3: Applying Redaction for {len(validated_entities)} Validated Entities ---")
                
                # Print the final, validated list for review
                for entity in sorted(validated_entities, key=lambda x: (x['page_num'], x['start_char'])):
                    score_info = f"(Score: {entity['score']:.2f})" if 'score' in entity else ""
                    print(f"  - Page {entity['page_num']+1} | Label: {entity['label']:<30} | Text: '{entity['text']}' {score_info}")
                
                redaction_success = redact_sensitive_info(input_pdf_path, validated_entities, output_pdf_path)

                if redaction_success:
                    logging.info("\nRedaction process completed successfully!")
                else:
                    logging.error("\nRedaction process failed.")
            else:
                logging.info("\nNo sensitive entities were validated for redaction after filtering.")
        else:
            logging.error("Failed to extract text from the PDF.")