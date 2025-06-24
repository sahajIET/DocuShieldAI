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

# --- OCR AUTO-TUNING ENGINE ---

# Define OCR parameter profiles for different document types
OCR_PROFILES = {
    "standard_scan": {
        "dpi": 400,
        "psm": 6,
        "denoise": True,
        "thresh_block_size": 29,
        "thresh_c": 5,
        "description": "Good for average quality office scans."
    },
    "high_quality_digital": {
        "dpi": 400,
        "psm": 6,
        "denoise": False, # Denoising not needed for clean images
        "thresh_block_size": 51, # Larger block size for uniform backgrounds
        "thresh_c": 10,
        "description": "Best for clean, high-contrast, digitally-born documents."
    },
    "noisy_or_low_contrast": {
        "dpi": 450, # Slightly higher DPI to resolve faint characters
        "psm": 6,
        "denoise": True,
        "thresh_block_size": 15, # Smaller block size for localized noise
        "thresh_c": 4,
        "description": "Optimized for faxes, photos of documents, or poor quality scans."
    }
}

def analyze_image_and_select_profile(image):
    """
    Analyzes a grayscale OpenCV image to determine its characteristics and
    selects the most appropriate OCR profile.

    Args:
        image (np.array): A grayscale image.

    Returns:
        dict: The selected OCR profile dictionary from OCR_PROFILES.
    """
    # 1. Estimate Contrast: Low standard deviation suggests low contrast.
    #    cv2.meanStdDev returns the mean and standard deviation of pixel intensities.
    mean, std_dev = cv2.meanStdDev(image)
    contrast = std_dev[0][0]

    # 2. Estimate Noise: The variance of the Laplacian can indicate noise level.
    #    A high variance suggests more edges, which often correlates with noise in scans.
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    # --- Define thresholds for selection (these can be fine-tuned) ---
    LOW_CONTRAST_THRESHOLD = 55.0
    HIGH_CONTRAST_THRESHOLD = 80.0
    HIGH_NOISE_THRESHOLD = 850.0

    # --- Profile Selection Logic ---
    if laplacian_var > HIGH_NOISE_THRESHOLD or contrast < LOW_CONTRAST_THRESHOLD:
        selected_profile = OCR_PROFILES["noisy_or_low_contrast"]
    elif contrast > HIGH_CONTRAST_THRESHOLD:
        selected_profile = OCR_PROFILES["high_quality_digital"]
    else:
        selected_profile = OCR_PROFILES["standard_scan"]

    print(f"    -> Analysis: Contrast={contrast:.2f}, Noise={laplacian_var:.2f}. " \
          f"Selected Profile: '{selected_profile['description']}'")

    return selected_profile


def extract_text_from_pdf_with_ocr(pdf_path):
    """
    Extracts text from a PDF, automatically selecting the best OCR profile for each page.
    """
    all_pages_data = []
    try:
        document = fitz.open(pdf_path)

        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            page_data = {
                'page_num': page_num, 'text': '', 'is_ocr_page': False, 'ocr_word_details': []
            }

            text_from_page = page.get_text("text")

            has_significant_images = len(page.get_images()) > 0
            is_text_sparse = len(text_from_page.strip()) < 100 # Increased threshold slightly

            if has_significant_images and is_text_sparse:
                page_data['is_ocr_page'] = True
                
                # --- AUTO-TUNING IN ACTION ---
                
                # 1. Render a temporary image for analysis
                temp_pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)) # 300 DPI is fine for analysis
                temp_img = Image.open(io.BytesIO(temp_pix.tobytes("png")))
                open_cv_image_gray = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2GRAY)
                
                # 2. Analyze the image to get the best profile
                profile = analyze_image_and_select_profile(open_cv_image_gray)
                
                # 3. Re-render the image with the optimal DPI from the selected profile
                pix = page.get_pixmap(matrix=fitz.Matrix(profile['dpi'] / 72, profile['dpi'] / 72))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                # --- IMAGE PREPROCESSING WITH SELECTED PROFILE ---
                final_gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                
                if profile['denoise']:
                    final_gray_image = cv2.fastNlMeansDenoising(final_gray_image, None, h=10)
                
                processed_image = cv2.adaptiveThreshold(
                    final_gray_image, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
                    profile['thresh_block_size'], profile['thresh_c']             
                )

                # --- Run Tesseract with the selected PSM ---
                ocr_data = pytesseract.image_to_data(
                    processed_image, 
                    output_type=pytesseract.Output.DATAFRAME,
                    config=f"--psm {profile['psm']}"
                )

                # (The rest of the function remains the same as your previous version)
                ocr_data = ocr_data.dropna(subset=['text'])
                ocr_data = ocr_data[ocr_data['text'].str.strip() != '']
                
                page_ocr_text = ""
                page_ocr_word_details = []
                current_char_offset = 0 

                for _, row in ocr_data.iterrows():
                    word = str(row['text'])
                    x0, y0, w, h = row['left'], row['top'], row['width'], row['height']
                    
                    scale_x = page.rect.width / pix.width
                    scale_y = page.rect.height / pix.height

                    word_rect = fitz.Rect(x0*scale_x, y0*scale_y, (x0+w)*scale_x, (y0+h)*scale_y)
                    
                    page_ocr_word_details.append({
                        'text': word, 'bbox': word_rect,
                        'start_char_in_ocr_text': current_char_offset,
                        'end_char_in_ocr_text': current_char_offset + len(word)
                    })
                    page_ocr_text += word + " " 
                    current_char_offset += len(word) + 1 

                page_data['text'] = page_ocr_text.strip()
                page_data['ocr_word_details'] = page_ocr_word_details
            else:
                print(f"  Page {page_num + 1}: Direct text extracted.")
                page_data['text'] = text_from_page
            
            all_pages_data.append(page_data)
        
        document.close()
        return all_pages_data

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
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
            
            print(f"--- Processing Page {page_num + 1} ---")
            for entity in page_entities:
                entity_text = entity["text"]
                is_ocr_entity = entity['is_ocr_page']

                if is_ocr_entity:
                    # --- NEW PRECISION LOGIC FOR OCR ---
                    ocr_word_details = entity.get('ocr_word_details')
                    if not ocr_word_details:
                        print(f"    - Warning: Skipping OCR entity '{entity_text}' due to missing word details.")
                        continue
                    
                    matched_bboxes = []
                    for word in ocr_word_details:
                        # Check if the character span of the OCR word overlaps with the entity's span
                        if (entity['start_char'] < word['end_char_in_ocr_text']) and \
                           (entity['end_char'] > word['start_char_in_ocr_text']):
                            matched_bboxes.append(word['bbox'])
                    
                    if matched_bboxes:
                        print(f"    - Redacting OCR entity '{entity_text}' with {len(matched_bboxes)} individual boxes.")
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
                         print(f"    - Warning: Could not map OCR entity '{entity_text}' to bounding boxes.")

                else: # This is a native text entity
                    print(f"    - Redacting native text entity '{entity_text}'.")
                    text_instances = page.search_for(entity_text)
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
            print(f"\nApplying {total_redactions_applied} redaction marks across the document...")
            for page in document:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, text=True)

            document.save(output_pdf_path, garbage=4, deflate=True) 
            print(f"Successfully applied redactions.")
            print(f"Redacted PDF saved to: {output_pdf_path}")
            return True
        else:
            print("\nNo sensitive entities were found or could be mapped to locations for redaction.")
            document.close()
            return False

    except Exception as e:
        print(f"An error occurred during redaction: {e}")
        return False


if __name__ == "__main__":
    print(f"Python Executable: {sys.executable}")
    try:
        print(f"PyMuPDF (fitz) version: {fitz.__version__}")
    except ImportError:
        print("PyMuPDF (fitz) is not imported or installed correctly.")

    input_pdf_path = "PLL_GDPR_Mixed_Paragraphs_Document.pdf"
    output_pdf_path = "redacted_document_auto_tuned.pdf"

    if not os.path.exists(input_pdf_path):
        print(f"Error: Input PDF file not found at '{input_pdf_path}'.")
    else:
        print(f"\n--- Starting Auto-Tuned Processing for: {input_pdf_path} ---")

        # Step 1: Extract text using the new auto-tuning function
        # No more manual parameters needed here!
        all_pages_data = extract_text_from_pdf_with_ocr(input_pdf_path)

        if all_pages_data:
            all_detected_entities = []

            print("\n--- Identifying Sensitive Entities (spaCy & Regex) ---")
            for page_data in all_pages_data:
                spacy_entities = identify_sensitive_entities([page_data])
                all_detected_entities.extend(spacy_entities)
                regex_entities = identify_sensitive_entities_regex(
                    page_data['text'], page_data['page_num'], page_data['is_ocr_page'], page_data.get('ocr_word_details')
                )
                all_detected_entities.extend(regex_entities)
            
            unique_entities = {
                (e['text'], e['label'], e['page_num'], e['start_char'], e['end_char']): e 
                for e in all_detected_entities
            }.values()
            
            if unique_entities:
                print(f"Detected {len(unique_entities)} unique sensitive entities.")
                # (The rest of the printing and redaction logic is the same)
                
                print("\n--- Applying Redaction ---")
                # Your excellent redact_sensitive_info function does the final, precise work
                redaction_success = redact_sensitive_info(input_pdf_path, list(unique_entities), output_pdf_path)

                if redaction_success:
                    print("\nRedaction process completed successfully!")
                else:
                    print("\nRedaction process failed.")
            else:
                print("No sensitive entities detected.")
        else:
            print("Failed to extract text from the PDF.")