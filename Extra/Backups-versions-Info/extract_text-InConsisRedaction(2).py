import fitz  # PyMuPDF is imported as 'fitz'
import pytesseract
from PIL import Image
import io
import spacy
import os
import sys
import pandas as pd # Will use pandas to parse tesseract output, install if not present

# --- IMPORTANT for Windows users: ---
# If tesseract.exe is not in your PATH, you might need to specify its path.
# For example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Uncomment the line below and replace with your actual path if needed.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ensure pandas is installed: pip install pandas

# Load the spaCy English language model
try:
    nlp = spacy.load("en_core_web_lg")
    print("spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please ensure you have run 'python -m spacy download en_core_web_lg'")
    nlp = None


def extract_text_from_pdf_with_ocr(pdf_path, dpi=300):
    """
    Extracts text from a PDF file, attempting OCR if direct text extraction is limited.
    Returns a list of dictionaries, one per page, with page_num, text, is_ocr_page,
    and optionally ocr_word_details (text and bounding boxes).

    Args:
        pdf_path (str): The file path to the PDF document.
        dpi (int): Dots per inch for rendering PDF pages to images for OCR.

    Returns:
        list: A list of dictionaries, each structured as:
              {
                  'page_num': int,
                  'text': str,
                  'is_ocr_page': bool,
                  'ocr_word_details': list of {'text': str, 'bbox': fitz.Rect} (only if is_ocr_page)
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

            # If direct extraction yields very little text (e.g., blank or image-only)
            if len(text_from_page.strip()) < 50:
                print(f"  Page {page_num + 1}: Little direct text found, attempting OCR...")
                page_data['is_ocr_page'] = True

                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                # Use image_to_data to get text along with bounding boxes
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)

                # Clean up data: remove empty words and convert relevant columns
                ocr_data = ocr_data.dropna(subset=['text'])
                ocr_data = ocr_data[ocr_data['text'].str.strip() != '']

                page_ocr_text = ""
                page_ocr_word_details = []

                for index, row in ocr_data.iterrows():
                    word = str(row['text'])
                    # Tesseract bounding boxes are (left, top, width, height)
                    # PyMuPDF Rect needs (x0, y0, x1, y1)
                    x0, y0, w, h = row['left'], row['top'], row['width'], row['height']
                    x1, y1 = x0 + w, y0 + h

                    # Scale OCR bbox from image pixels to PDF page coordinates
                    # PyMuPDF page dimensions
                    page_width = page.rect.width
                    page_height = page.rect.height

                    # Image dimensions after rendering with dpi
                    img_width_px = pix.width
                    img_height_px = pix.height

                    # Calculate scaling factors
                    scale_x = page_width / img_width_px
                    scale_y = page_height / img_height_px

                    # Apply scaling to get PDF coordinates
                    pdf_x0 = x0 * scale_x
                    pdf_y0 = y0 * scale_y
                    pdf_x1 = x1 * scale_x
                    pdf_y1 = y1 * scale_y

                    word_rect = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)

                    page_ocr_word_details.append({'text': word, 'bbox': word_rect})
                    page_ocr_text += word + " " # Reconstruct text for spaCy, add space after each word

                page_data['text'] = page_ocr_text.strip()
                page_data['ocr_word_details'] = page_ocr_word_details
                print("\nNote: OCR was used for page " + str(page_num + 1) + ". Accuracy may vary.") # Specific page note
            else:
                print(f"  Page {page_num + 1}: Direct text extracted.")
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

    # Define categories of PII/SPI we are interested in.
    PII_LABELS = ["PERSON", "NORP", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "MONEY", "TIME", "FAC"] # Added FAC

    for page_data in pages_data:
        page_text = page_data['text']
        page_num = page_data['page_num']
        is_ocr_page = page_data['is_ocr_page']
        ocr_word_details = page_data.get('ocr_word_details', []) # Get, with default empty list

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
                ocr_word_details_for_page = next((entity['ocr_word_details'] for entity in page_entities if entity['is_ocr_page']), [])
                
                if not ocr_word_details_for_page:
                    print(f"    Warning: Page {page_num + 1} marked as OCR but no word details found. Skipping OCR redaction.")
                    continue

                for entity in page_entities:
                    matched_bboxes = []
                    entity_words = set(entity['text'].lower().split())

                    for ocr_word_detail in ocr_word_details_for_page:
                        if ocr_word_detail['text'].lower() in entity_words:
                            matched_bboxes.append(ocr_word_detail['bbox'])
                    
                    if matched_bboxes:
                        combined_bbox = fitz.Rect(matched_bboxes[0])
                        for bbox in matched_bboxes[1:]:
                            combined_bbox.include_rect(bbox)
                        
                        page.add_redact_annot(combined_bbox, fill=(0, 0, 0))
                        redaction_count += 1
                    else:
                        print(f"    Warning: Could not find OCR bounding box for entity '{entity['text']}' on page {page_num + 1}.")
            else:
                print(f"  Page {page_num + 1}: Applying redaction using native text search.")
                for entity in page_entities:
                    entity_text = entity["text"]
                    text_instances = page.search_for(entity_text)
                    for inst in text_instances:
                        # --- THIS IS THE FIX ---
                        # Instead of inst.expand(2), directly modify the coordinates.
                        inst.x0 -= 2  # Expand left by 2 units
                        inst.y0 -= 2  # Expand top by 2 units
                        inst.x1 += 2  # Expand right by 2 units
                        inst.y1 += 2  # Expand bottom by 2 units
                        
                        page.add_redact_annot(inst,text="█", fill=(0, 0, 0))
                        redaction_count += 1

        if redaction_count > 0:
            for page in document:
                # The has_redact_annot() method was removed in recent versions.
                # apply_redactions() will simply do nothing if there are no annotations.
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

            document.save(output_pdf_path, garbage=4, deflate=True) # garbage=4 is a good value
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
    input_pdf_path = "PLL5page.pdf" 
    output_pdf_path = "redacted_document1.pdf"

    if not os.path.exists(input_pdf_path):
        print(f"Error: Input PDF file not found at '{input_pdf_path}'.")
        print("Please create or place your sample PDF (with PII) in the project directory.")
    else:
        print(f"\n--- Starting processing for: {input_pdf_path} ---")

        # Step 1: Extract text from PDF (with OCR support), getting detailed page data
        # This returns a list of dictionaries, one for each page.
        all_pages_data = extract_text_from_pdf_with_ocr(input_pdf_path)

        if all_pages_data:
            print("\n--- Identifying Sensitive Entities ---")
            # Step 2: Identify sensitive entities using spaCy, leveraging page data
            detected_entities = identify_sensitive_entities(all_pages_data)

            if detected_entities:
                print(f"Detected {len(detected_entities)} sensitive entities:")
                # Group and print by page for better readability
                for page_num in sorted(list(set(e['page_num'] for e in detected_entities))):
                    print(f"  Page {page_num + 1}:")
                    for entity in [e for e in detected_entities if e['page_num'] == page_num]:
                        print(f"    Text: '{entity['text']}', Label: {entity['label']}, OCR: {entity['is_ocr_page']}")

                print("\n--- Applying Redaction ---")
                # Step 3: Redact sensitive information in the PDF, now passing all detected entities
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