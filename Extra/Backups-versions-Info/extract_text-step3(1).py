import fitz  # PyMuPDF is imported as 'fitz'
import pytesseract
from PIL import Image
import io
import spacy # Import spacy

# --- IMPORTANT for Windows users: ---
# If tesseract.exe is not in your PATH, you might need to specify its path.
# For example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Uncomment the line below and replace with your actual path if needed.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the spaCy English language model
# This needs to be done once when the application starts
try:
    nlp = spacy.load("en_core_web_lg")
    print("spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Please ensure you have run 'python -m spacy download en_core_web_lg'")
    nlp = None # Set nlp to None if loading fails


def extract_text_from_pdf_with_ocr(pdf_path, dpi=300):
    """
    Extracts text from a PDF file, attempting OCR if direct text extraction is limited.

    Args:
        pdf_path (str): The file path to the PDF document.
        dpi (int): Dots per inch for rendering PDF pages to images for OCR.

    Returns:
        str: A single string containing all extracted text,
             or None if an error occurs.
    """
    try:
        document = fitz.open(pdf_path)
        all_extracted_text = ""
        ocr_used_flag = False # Rename to avoid conflict if 'ocr_used' is global in mind.

        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text_from_page = page.get_text("text")

            if len(text_from_page.strip()) < 50:
                # Heuristic: If little text and no embedded images (likely blank or scanned)
                print(f"  Page {page_num + 1}: Little direct text found, attempting OCR...")
                ocr_used_flag = True
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img)
                all_extracted_text += ocr_text + "\n"
            else:
                print(f"  Page {page_num + 1}: Direct text extracted.")
                all_extracted_text += text_from_page + "\n"

        document.close()

        if ocr_used_flag:
            print("\nNote: OCR was used for one or more pages. Accuracy may vary.")

        return all_extracted_text

    except Exception as e:
        print(f"An error occurred during PDF processing with OCR: {e}")
        return None

def identify_sensitive_entities(text):
    """
    Identifies sensitive entities in the given text using spaCy's NER.

    Args:
        text (str): The input text to analyze.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected entity
              with 'text', 'label', 'start_char', and 'end_char'.
    """
    if nlp is None:
        print("NLP model not loaded, cannot identify entities.")
        return []

    doc = nlp(text)
    sensitive_entities = []

    # Define categories of PII/SPI we are interested in.
    # spaCy's pre-trained 'en_core_web_lg' model detects many common types.
    # You can customize this list based on what constitutes 'sensitive' for your project.
    PII_LABELS = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", 
                  "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", 
                  "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]

    # For a basic project, we might focus on:
    # PERSON: People's names, including fictional.
    # GPE: Countries, cities, states.
    # ORG: Companies, agencies, institutions.
    # DATE: Absolute or relative dates or periods.
    # CARDINAL: Numerals that do not fall under other types (can often be part of IDs, phone numbers)
    # Other labels might be contextually sensitive depending on your specific definition of PII.

    for ent in doc.ents:
        if ent.label_ in PII_LABELS:
            sensitive_entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
    return sensitive_entities


if __name__ == "__main__":
    # --- IMPORTANT: Replace with the actual name of your PDF file ---
    # Use a PDF that has some names, dates, places, etc., for testing NER.
    pdf_to_process = "PLL.pdf" 
    # Make sure this file exists in your project directory.
    # If you don't have one, use your sample.pdf or sample_scanned.pdf and add some PII.

    print(f"\n--- Starting processing for: {pdf_to_process} ---")
    extracted_content = extract_text_from_pdf_with_ocr(pdf_to_process)

    if extracted_content:
        print("\n--- Identifying Sensitive Entities ---")
        detected_entities = identify_sensitive_entities(extracted_content)

        if detected_entities:
            print(f"Detected {len(detected_entities)} sensitive entities:")
            for entity in detected_entities:
                print(f"  Text: '{entity['text']}', Label: {entity['label']}, "
                      f"Start: {entity['start_char']}, End: {entity['end_char']}")
        else:
            print("No sensitive entities detected using spaCy's pre-trained model.")
    else:
        print("Failed to extract text, skipping entity identification.")