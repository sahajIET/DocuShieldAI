import fitz  # PyMuPDF is imported as 'fitz'
import pytesseract
from PIL import Image
import io
import spacy
import os
import sys
import pandas as pd
import numpy as np  # Import NumPy for image conversion
import cv2  # Import the OpenCV library
import re   # For regular expressions

# --- IMPORTANT for Windows users: ---
# If tesseract.exe is not in your PATH, you might need to specify its path.
# For example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Uncomment the line below and replace with your actual path if needed.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the spaCy English language model
try:
    nlp = spacy.load("en_core_web_lg")
    print("spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please ensure you have run 'python -m spacy download en_core_web_lg'")
    nlp = None


def extract_text_from_pdf_with_ocr(pdf_path, dpi=300, 
                                 tesseract_psm=6, # Default to PSM 6
                                 adaptive_thresh_block_size=11, 
                                 adaptive_thresh_c=2,
                                 apply_denoising=True):
    """
    Extracts text from a PDF file, attempting OCR if direct text extraction is limited.
    Returns a list of dictionaries, one per page, with page_num, text, is_ocr_page,
    and optionally ocr_word_details (text, bounding box, and char offsets in reconstructed text).

    Args:
        pdf_path (str): The file path to the PDF document.
        dpi (int): Dots per inch for rendering PDF pages to images for OCR.
        tesseract_psm (int): Tesseract Page Segmentation Mode (PSM).
                             Common values: 3 (default), 6 (assume single block), 7 (single text line), 8 (single word).
        adaptive_thresh_block_size (int): Size of a pixel neighborhood for adaptive thresholding. Must be odd.
        adaptive_thresh_c (int): Constant subtracted from the mean or weighted mean.
        apply_denoising (bool): Whether to apply non-local means denoising.

    Returns:
        list: A list of dictionaries, each structured as:
              {
                  'page_num': int,
                  'text': str,
                  'is_ocr_page': bool,
                  'ocr_word_details': list of {'text': str, 'bbox': fitz.Rect, 'start_char_in_ocr_text': int, 'end_char_in_ocr_text': int} (only if is_ocr_page)
              }
              Returns an empty list if an error occurs.
    """
    all_pages_data = []
    try:
        document = fitz.open(pdf_path)

        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            page_data = {
                'page_num': page_num,
                'text': '',
                'is_ocr_page': False,
                'ocr_word_details': []
            }

            text_from_page = page.get_text("text")

            # Determine if this page should be OCR'd:
            # Apply OCR ONLY IF the page has raster images AND direct text extracted is sparse.
            is_text_sparse = len(text_from_page.strip()) < 50
            has_raster_images = len(page.get_images()) > 0 

            # --- REVISED OCR TRIGGERING LOGIC ---
            if has_raster_images and is_text_sparse:
                print(f"  Page {page_num + 1}: Detected as image-heavy with sparse direct text. Attempting OCR...")
                page_data['is_ocr_page'] = True

                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                # --- IMAGE PREPROCESSING WITH OPENCV ---
                open_cv_image = np.array(img.convert('RGB')) 
                gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY) 
                
                if apply_denoising:
                    gray_image = cv2.fastNlMeansDenoising(
                        gray_image, 
                        None, 
                        h=10, 
                        templateWindowSize=7, 
                        searchWindowSize=21
                    )
                
                processed_image = cv2.adaptiveThreshold(
                    gray_image, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 
                    adaptive_thresh_block_size, 
                    adaptive_thresh_c             
                )
                # --- END OF IMAGE PREPROCESSING ---

                # Use image_to_data to get text along with bounding boxes
                tesseract_config_str = f"--psm {tesseract_psm}" 
                ocr_data = pytesseract.image_to_data(
                    processed_image, 
                    output_type=pytesseract.Output.DATAFRAME,
                    config=tesseract_config_str
                )

                ocr_data = ocr_data.dropna(subset=['text'])
                ocr_data = ocr_data[ocr_data['text'].str.strip() != '']
                
                page_ocr_text = ""
                page_ocr_word_details = []
                current_char_offset = 0 # Track current char offset in reconstructed text

                for index, row in ocr_data.iterrows():
                    word = str(row['text'])
                    
                    x0, y0, w, h = row['left'], row['top'], row['width'], row['height']
                    x1, y1 = x0 + w, y0 + h

                    page_width = page.rect.width
                    page_height = page.rect.height
                    
                    img_width_px = pix.width
                    img_height_px = pix.height
                    
                    scale_x = page_width / img_width_px
                    scale_y = page_height / img_height_px

                    pdf_x0 = x0 * scale_x
                    pdf_y0 = y0 * scale_y
                    pdf_x1 = x1 * scale_x
                    pdf_y1 = y1 * scale_y

                    word_rect = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
                    
                    # --- NEW: Store character offsets in reconstructed OCR text ---
                    word_start_char = current_char_offset
                    word_end_char = current_char_offset + len(word)
                    
                    page_ocr_word_details.append({
                        'text': word, 
                        'bbox': word_rect,
                        'start_char_in_ocr_text': word_start_char,
                        'end_char_in_ocr_text': word_end_char
                    })
                    page_ocr_text += word + " " 
                    current_char_offset += len(word) + 1 # +1 for the space added

                page_data['text'] = page_ocr_text.strip()
                page_data['ocr_word_details'] = page_ocr_word_details
                print("Note: OCR was used for page " + str(page_num + 1) + ". Accuracy may vary.") 
            else:
                print(f"  Page {page_num + 1}: Direct text extracted (no significant images or sufficient text).")
                page_data['text'] = text_from_page
            
            all_pages_data.append(page_data)
        
        document.close()
        return all_pages_data

    except Exception as e:
        print(f"An error occurred during PDF processing with OCR: {e}")
        return []


def identify_sensitive_entities(pages_data):
    """
    Identifies sensitive entities across all pages using spaCy's NER,
    and includes page-specific context.

    Args:
        pages_data (list): List of dictionaries, each from extract_text_from_pdf_with_ocr.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected entity
              with 'text', 'label', 'page_num', 'start_char', 'end_char',
              'is_ocr_page', and 'ocr_word_details' (if applicable for that page).
    """
    if nlp is None:
        print("NLP model not loaded, cannot identify entities.")
        return []
        
    all_sensitive_entities = []
    
    PII_LABELS = ["PERSON", "NORP", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "MONEY", "TIME", "FAC"]

    for page_data in pages_data:
        page_text = page_data['text']
        page_num = page_data['page_num']
        is_ocr_page = page_data['is_ocr_page']
        ocr_word_details = page_data.get('ocr_word_details', []) 

        doc = nlp(page_text)
        
        for ent in doc.ents:
            if ent.label_ in PII_LABELS:
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                    "page_num": page_num,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "is_ocr_page": is_ocr_page,
                    "ocr_word_details": ocr_word_details # Attach page's OCR word details directly
                }
                all_sensitive_entities.append(entity_info)
    return all_sensitive_entities

def identify_sensitive_entities_regex(text, page_num, is_ocr_page, ocr_word_details=None):
    """
    Identifies sensitive entities using regular expressions.
    """
    regex_entities = []

    patterns = {
        "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE_NUMBER": r"(\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", 
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b", 
        "SSN_US": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", 
        "AADHAAR_ID": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
        "PAN_ID": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        "CUSTOM_KEYWORD_PLL": r"\bPLL\b|\b[Pp][Ll][Ll]\b" 
    }

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            regex_entities.append({
                "text": match.group(0),
                "label": label,
                "page_num": page_num,
                "start_char": match.start(),
                "end_char": match.end(),
                "is_ocr_page": is_ocr_page,
                "ocr_word_details": ocr_word_details # Pass along for consistent handling
            })
    return regex_entities


def redact_sensitive_info(pdf_path, detected_entities, output_pdf_path):
    """
    Redacts sensitive information in a PDF based on detected entities.
    Handles both native text and OCR-based pages.

    Args:
        pdf_path (str): Path to the input PDF file.
        detected_entities (list): List of dictionaries, each with 'text', 'label', 'page_num',
                                  'start_char', 'end_char', 'is_ocr_page', 'ocr_word_details'.
        output_pdf_path (str): Path to save the redacted PDF file.

    Returns:
        bool: True if redaction was successful, False otherwise.
    """
    try:
        document = fitz.open(pdf_path)
        redaction_count = 0

        # Group entities by page for easier processing
        entities_by_page = {}
        for entity in detected_entities:
            page_num = entity['page_num']
            if page_num not in entities_by_page:
                entities_by_page[page_num] = []
            entities_by_page[page_num].append(entity)

        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            page_entities = entities_by_page.get(page_num, [])

            if not page_entities:
                continue

            page_was_ocrd = any(entity['is_ocr_page'] for entity in page_entities)

            if page_was_ocrd:
                print(f"  Page {page_num + 1}: Applying redaction using OCR bounding boxes.")
                # Retrieve the full list of OCR word details for this page
                ocr_word_details_for_page = None
                for entity in page_entities:
                    if entity['is_ocr_page'] and entity.get('ocr_word_details'):
                        ocr_word_details_for_page = entity['ocr_word_details']
                        break 
                
                if not ocr_word_details_for_page:
                    print(f"    Warning: Page {page_num + 1} marked as OCR but no word details found. Skipping OCR redaction.")
                    continue

                for entity in page_entities:
                    entity_start = entity['start_char']
                    entity_end = entity['end_char']
                    matched_bboxes = []
                    
                    # Iterate through OCR word details and check for character offset overlap
                    for ocr_word_detail in ocr_word_details_for_page:
                        word_start = ocr_word_detail['start_char_in_ocr_text']
                        word_end = ocr_word_detail['end_char_in_ocr_text']
                        
                        # Check for overlap: (start_A < end_B) and (end_A > start_B)
                        if (entity_start < word_end) and (entity_end > word_start):
                            matched_bboxes.append(ocr_word_detail['bbox'])
                    
                    if matched_bboxes:
                        combined_bbox = fitz.Rect(matched_bboxes[0])
                        for bbox in matched_bboxes[1:]:
                            combined_bbox.include_rect(bbox)
                        
                        # Expand slightly to ensure full coverage
                        margin = 2
                        combined_bbox = fitz.Rect(
                            combined_bbox.x0 - margin,
                            combined_bbox.y0 - margin,
                            combined_bbox.x1 + margin,
                            combined_bbox.y1 + margin
                        )
                        # Ensure the expanded rectangle is still within page bounds
                        combined_bbox = combined_bbox.intersect(page.rect)

                        page.add_redact_annot(combined_bbox, fill=(0, 0, 0))
                        redaction_count += 1
                    else:
                        print(f"    Warning: Could not find OCR bounding box for entity '{entity['text']}' (chars {entity_start}-{entity_end}) on page {page_num + 1}.")
            else:
                print(f"  Page {page_num + 1}: Applying redaction using native text search.")
                for entity in page_entities:
                    entity_text = entity["text"]
                    text_instances = page.search_for(entity_text)
                    for inst in text_instances:
                        # Manually expand the rectangle to cover whole words
                        margin = 2 
                        expanded_rect = fitz.Rect(
                            inst.x0 - margin,
                            inst.y0 - margin,
                            inst.x1 + margin,
                            inst.y1 + margin
                        )
                        # Ensure the expanded rectangle is still within page bounds
                        expanded_rect = expanded_rect.intersect(page.rect)
                        
                        page.add_redact_annot(expanded_rect,text="█", fill=(0, 0, 0))
                        redaction_count += 1

        if redaction_count > 0:
            for page in document:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

            document.save(output_pdf_path, garbage=4, deflate=True) 
            print(f"Successfully redacted {redaction_count} instances.")
            print(f"Redacted PDF saved to: {output_pdf_path}")
            return True
        else:
            print("No sensitive entities found on pages to redact.")
            document.close()
            return False

    except Exception as e:
        print(f"An error occurred during redaction: {e}")
        return False


if __name__ == "__main__":
    # Python Executable and PyMuPDF version diagnostics
    print(f"Python Executable: {sys.executable}")
    try:
        import fitz
        print(f"PyMuPDF (fitz) version: {fitz.__version__}")
    except ImportError:
        print("PyMuPDF (fitz) is not imported or installed correctly.")

    # --- IMPORTANT: Configure your input and output PDF paths ---
    input_pdf_path = "GDPR_PII_Mixed_Document.pdf" 
    output_pdf_path = "redacted_document_ocr_test.pdf" 

    if not os.path.exists(input_pdf_path):
        print(f"Error: Input PDF file not found at '{input_pdf_path}'.")
        print("Please create or place your sample PDF (with PII) in the project directory.")
    else:
        print(f"\n--- Starting processing for: {input_pdf_path} ---")

        all_pages_data = extract_text_from_pdf_with_ocr(
            input_pdf_path, 
            dpi=300, # Common starting point, increase for higher quality images
            tesseract_psm=6, # Assumes a single uniform block of text. Try 3 for general, 7 for single line.
            adaptive_thresh_block_size=15, # Must be odd. Affects local thresholding.
            adaptive_thresh_c=5, # Constant subtracted. Adjust for contrast.
            apply_denoising=True # Toggle for noisy images
        )

        if all_pages_data:
            print("\n--- Identifying Sensitive Entities ---")
            all_detected_entities = []
            for page_data in all_pages_data:
                spacy_entities_on_page = identify_sensitive_entities([page_data])
                all_detected_entities.extend(spacy_entities_on_page)

                regex_entities_on_page = identify_sensitive_entities_regex(
                    page_data['text'],
                    page_data['page_num'],
                    page_data['is_ocr_page'],
                    page_data.get('ocr_word_details')
                )
                all_detected_entities.extend(regex_entities_on_page)

            # Deduplicate entities 
            unique_entities = []
            seen = set()
            for entity in all_detected_entities:
                entity_tuple = (entity['text'], entity['label'], entity['page_num'], 
                                entity['start_char'], entity['end_char'])
                if entity_tuple not in seen:
                    unique_entities.append(entity)
                    seen.add(entity_tuple)
            detected_entities = unique_entities

            if detected_entities:
                print(f"Detected {len(detected_entities)} unique sensitive entities:")
                for page_num_print in sorted(list(set(e['page_num'] for e in detected_entities))):
                    print(f"  Page {page_num_print + 1}:")
                    for entity in [e for e in detected_entities if e['page_num'] == page_num_print]:
                        print(f"    Text: '{entity['text']}', Label: {entity['label']}, OCR: {entity['is_ocr_page']}")

                print("\n--- Applying Redaction ---")
                redaction_success = redact_sensitive_info(input_pdf_path, detected_entities, output_pdf_path)

                if redaction_success:
                    print("\nRedaction process completed successfully!")
                    print(f"Check '{output_pdf_path}' for the redacted document.")
                else:
                    print("\nRedaction process failed or no entities were actually redacted.")
            else:
                print("No sensitive entities detected using spaCy's pre-trained model.")
                print("No redaction applied.")
        else:
            print("Failed to extract text from the PDF, cannot proceed with redaction.")