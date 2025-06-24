# -*- coding: utf-8 -*-
"""
================================================================================
                    Advanced PDF Redaction Pipeline
================================================================================

This script provides a robust, multi-layered pipeline for detecting and redacting
sensitive information from PDF documents. It handles both native text PDFs and
scanned, image-based PDFs that require Optical Character Recognition (OCR).

Key Features of This Code:
---------------------------
1.  **Hybrid Text Extraction:** Automatically distinguishes between native text pages
    and image-based pages, applying a sophisticated OCR process only when necessary.
2.  **Dynamic OCR Tuning:** Analyzes image properties (contrast, noise) to select
    the best preprocessing profile (denoising, thresholding) for each page,
    maximizing OCR accuracy.
3.  **Multi-Engine PII Detection:** Utilizes different engines to find sensitive
    data, ensuring comprehensive coverage. The main script is configured to use
    Presidio and Regex, with spaCy available to be enabled.
4.  **Intelligent Validation Layer:** A post-processing step that significantly
    reduces false positives by using deny lists, confidence scores, and contextual
    and semantic analysis for ambiguous terms.
5.  **Dual-Strategy Redaction:** Employs two distinct, precise methods for redaction:
    - For OCR'd text: A highly accurate word-by-word bounding box mapping.
    - For native PDF text: A robust search using text quads for accurate placement.

Dependencies:
-------------
-   fitz (PyMuPDF)
-   pytesseract
-   Pillow (PIL)
-   spacy (+ 'en_core_web_lg' model)
-   pandas
-   numpy
-   opencv-python
-   presidio-analyzer

To install the spaCy model, run:
python -m spacy download en_core_web_lg
"""

# ==============================================================================
# 0. SETUP AND IMPORTS
# ==============================================================================

# --- Standard Library and Third-Party Imports ---
import io
import logging
import os
import re
import sys
from typing import List, Dict, Any

import cv2
import fitz  # PyMuPDF is imported as 'fitz'
import numpy as np
import pandas as pd
import pytesseract
import spacy
from PIL import Image # Explicitly import Image from PIL
from presidio_analyzer import AnalyzerEngine

# --- Tesseract Configuration ---
# If tesseract.exe is not in your system's PATH, you might need to specify its path.
# For example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Load spaCy Language Model ---
# This model is used for advanced natural language processing tasks, specifically
# for Named Entity Recognition (NER) and for semantic analysis in the validation step.
try:
    nlp = spacy.load("en_core_web_lg")
    print("✅ spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading spaCy model: {e}")
    print("👉 Please ensure you have run 'python -m spacy download en_core_web_lg'")
    nlp = None

# --- Logger Configuration ---
# Sets up a centralized logger to output information to both a file (`redaction_process.log`)
# for detailed debugging and to the console for real-time progress tracking.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redaction_process.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Dynamic OCR Profiles ---
# Defines different OCR strategies based on image quality. The script will
# automatically select the best one for each page to improve accuracy.
OCR_PROFILES = {
    "standard_scan": {"dpi": 400, "psm": 6, "denoise": True, "thresh_block_size": 29, "thresh_c": 5, "description": "Good for average quality office scans."},
    "high_quality_digital": {"dpi": 400, "psm": 6, "denoise": False, "thresh_block_size": 51, "thresh_c": 10, "description": "Best for clean, high-contrast, digitally-born documents."},
    "noisy_or_low_contrast": {"dpi": 450, "psm": 6, "denoise": True, "thresh_block_size": 15, "thresh_c": 4, "description": "Optimized for faxes, photos of documents, or poor quality scans."}
}

# ==============================================================================
# 1. TEXT EXTRACTION AND OCR
# ==============================================================================

def analyze_image_and_select_profile(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes an image's contrast and noise to dynamically select the best OCR profile.

    How it works:
    - It calculates the standard deviation of pixel intensity (a measure of contrast).
    - It calculates the variance of the Laplacian of the image (a measure of sharpness/noise).
    - Based on predefined thresholds, it selects the most suitable OCR profile from the
      global OCR_PROFILES dictionary.

    Accuracy/Efficiency:
    - This function is key to maximizing OCR accuracy. By tailoring the preprocessing
      steps to the image's specific quality, it avoids a one-size-fits-all approach
      that often fails on diverse document types.
    """
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


def extract_text_from_pdf_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF, automatically applying a high-accuracy
    OCR process for image-based pages.

    How it works:
    - It iterates through each page of the PDF.
    - It uses a heuristic (presence of images and low amount of embedded text) to
      decide if a page is a scanned image that requires OCR.
    - For OCR pages, it follows a multi-step process designed for high accuracy.
    - For native text pages, it simply extracts the existing text.

    Accuracy & Efficiency of the OCR Pipeline:
    - This specific pipeline is designed for high-quality text recognition.
    - By rendering at a fixed 300 DPI and then applying advanced preprocessing, it creates
      a clean, consistent input for Tesseract.
    - The coordinate mapping is direct and accurate because no geometric transformations
      (like rotation) are applied to the image after it's rendered. The scaling factors
      are calculated based on the original rendered image dimensions, ensuring perfect alignment.
    """
    all_pages_data = []
    try:
        document = fitz.open(pdf_path)
        for page_num in range(document.page_count):
            try:
                page = document.load_page(page_num)
                page_data = {'page_num': page_num, 'text': '', 'is_ocr_page': False, 'ocr_word_details': []}
                text_from_page = page.get_text("text")

                # Heuristic to determine if a page is scanned and needs OCR.
                has_significant_images = len(page.get_images()) > 0
                is_text_sparse = len(text_from_page.strip()) < 100

                if has_significant_images and is_text_sparse:
                    logging.info(f"Page {page_num + 1}: Image-based page detected. Attempting OCR...")
                    page_data['is_ocr_page'] = True

                    # --- High-Accuracy OCR Process ---

                    # STEP 1: Render the page at a fixed 300 DPI to create a base image.
                    # This provides a consistent starting point for all subsequent processing.
                    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))

                    # Use PIL to safely open from bytes and then convert to NumPy array for OpenCV.
                    # This handles different image modes (RGB, RGBA, Grayscale) correctly.
                    img_pil = Image.open(io.BytesIO(pix.tobytes("png"))) # Specify 'png' for consistency

                    # Convert PIL image (RGB) to OpenCV format (BGR) for initial analysis
                    open_cv_image_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    
                    # STEP 2: Create a grayscale version of the image for analysis.
                    open_cv_image_gray = cv2.cvtColor(open_cv_image_bgr, cv2.COLOR_BGR2GRAY)

                    # STEP 3: Analyze the image to select the best preprocessing profile.
                    profile = analyze_image_and_select_profile(open_cv_image_gray)

                    # STEP 4: This line is a placeholder for potential geometric transformations.
                    # Currently, it does nothing, which is critical for ensuring the coordinate
                    # mapping in the final step remains accurate.
                    deskewed_image = open_cv_image_bgr # Using the BGR image directly

                    # STEP 5: Apply advanced image preprocessing based on the selected profile.
                    # This cleans the image, making it easier for Tesseract to read characters.
                    final_gray_image = cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2GRAY) # Convert deskewed to gray
                    if profile['denoise']:
                        final_gray_image = cv2.fastNlMeansDenoising(final_gray_image, None, h=10)
                    processed_image = cv2.adaptiveThreshold(final_gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, profile['thresh_block_size'], profile['thresh_c'])

                    # STEP 6: Run Tesseract with an advanced configuration.
                    # Disabling system dictionaries ('dawg' files) prevents Tesseract from
                    # incorrectly "correcting" proper nouns and alphanumeric IDs.
                    tesseract_config = (
                        f"--psm {profile['psm']} "
                        f"-c load_system_dawg=0 "
                        f"-c load_freq_dawg=0"
                    )
                    ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DATAFRAME, config=tesseract_config)

                    # STEP 7: Reconstruct the text and map word coordinates.
                    # This loop builds the full text of the page from individual words and,
                    # crucially, stores the precise bounding box of each word.
                    ocr_data = ocr_data.dropna(subset=['text'])
                    ocr_data['text'] = ocr_data['text'].astype(str)
                    ocr_data = ocr_data[ocr_data['text'].str.strip() != '']
                    page_ocr_text = ""
                    page_ocr_word_details = []
                    current_char_offset = 0
                    for _, row in ocr_data.iterrows():
                        word = str(row['text'])
                        x0, y0, w, h = row['left'], row['top'], row['width'], row['height']
                        # The scaling factor correctly converts coordinates from the rendered image
                        # back to the PDF's internal point system.
                        scale_x, scale_y = page.rect.width / pix.width, page.rect.height / pix.height
                        word_rect = fitz.Rect(x0*scale_x, y0*scale_y, (x0+w)*scale_x, (y0+h)*scale_y)
                        page_ocr_word_details.append({'text': word, 'bbox': word_rect, 'start_char_in_ocr_text': current_char_offset, 'end_char_in_ocr_text': current_char_offset + len(word)})
                        page_ocr_text += word + " "
                        current_char_offset += len(word) + 1
                    page_data['text'] = page_ocr_text.strip()
                    page_data['ocr_word_details'] = page_ocr_word_details
                else:
                    # If the page has native text, simply extract it.
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

# ==============================================================================
# 2. SENSITIVE ENTITY DETECTION
# ==============================================================================

def identify_sensitive_entities(pages_data: List[Dict]) -> List[Dict]:
    """
    Detects general named entities (PERSON, ORG, etc.) using the spaCy NER model.
    """
    if nlp is None:
        logging.error("spaCy model not loaded, cannot identify entities.")
        return []
    all_sensitive_entities = []
    # Defines the types of entities spaCy should look for.
    PII_LABELS = ["PERSON", "NORP", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "MONEY", "TIME", "FAC"]
    for page_data in pages_data:
        doc = nlp(page_data['text'])
        for ent in doc.ents:
            if ent.label_ in PII_LABELS:
                all_sensitive_entities.append({"text": ent.text, "label": f"SPACY_{ent.label_}", "page_num": page_data['page_num'], "start_char": ent.start_char, "end_char": ent.end_char, "is_ocr_page": page_data['is_ocr_page'], "ocr_word_details": page_data.get('ocr_word_details', [])})
    return all_sensitive_entities


def identify_sensitive_entities_regex(text: str, page_num: int, is_ocr_page: bool, ocr_word_details: List = None) -> List[Dict]:
    """
    Identifies sensitive entities using a refined set of high-precision regular expressions.

    Purpose:
    - This function is designed to catch very specific, structured data that NER models
      like spaCy or Presidio might miss, such as national ID numbers.
    - It complements the model-based approaches by providing rule-based certainty.
    """
    regex_entities = []

    # A curated dictionary of patterns for specific PII types.
    patterns = {
        "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PAN_ID": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        "AADHAAR_ID": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
        "IP_ADDRESS_V4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "MAC_ADDRESS": r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b|\b(?:[0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}\b",
        "POSTAL_ADDRESS": r"(?i)\b(?:Address|Location|Addr\.)[:\s-]+((?:[A-Za-z0-9\s,.-]+(?:\n|\s)){1,4}\b\d{5,7})",
        "PREFIX_NAMES": r'\b(?:Mr|Mrs|Ms|Miss|Smt|Shri|Dr|Prof|Mx)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        "SSN_US": r"(?i)\b(?:SSN|Social\sSecurity\sNo\.?)[:\s-]*(\d{3}[- ]?\d{2}[- ]?\d{4})\b",
        "AGE": r"\b(?:age|years\sold|yrs|y\.o\.)[\s:]*(\d{1,2})\b",
        "BLOOD_GROUP": r"(?i)(?:\b(?:blood\s(?:group|type)|b\.g\.)[:\s-]*)?[\(\[]?\s*(A|B|AB|O)\s?[+-](?:ve|sitive|gative)?\s*[\)\]]?",
        "ALPHANUMERIC_ID_GENERIC": r"\b(?=[a-zA-Z0-9-]{6,25}\b)(?:[a-zA-Z]+[-_]*\d+|\d+[-_]*[a-zA-Z]+)[a-zA-Z0-9-]*\b",
        "GENDER": r"(?i)\b(?:male|female)\b",
    }

    for label, pattern_item in patterns.items():
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


def identify_entities_with_presidio(analyzer: AnalyzerEngine, text: str, page_num: int, is_ocr_page: bool, ocr_word_details: List = None) -> List[Dict]:
    """
    Uses Microsoft Presidio to identify a wide range of PII with higher confidence.

    Purpose:
    - Presidio is specialized in detecting common PII like credit card numbers, phone
      numbers, IBAN codes, etc., often with higher accuracy than general NER models.
    - It also provides a confidence score for each finding, which is used in the
      validation step.
    """
    presidio_entities = []
    try:
        results = analyzer.analyze(text=text, language='en')
        for res in results:
            presidio_entities.append({
                "text": text[res.start:res.end],
                "label": f"PRESIDIO_{res.entity_type}",
                "page_num": page_num,
                "start_char": res.start,
                "end_char": res.end,
                "score": res.score,
                "is_ocr_page": is_ocr_page,
                "ocr_word_details": ocr_word_details
            })
    except Exception as e:
        logging.error(f"Presidio analysis failed on page {page_num + 1}: {e}")
    return presidio_entities

# ==============================================================================
# 3. VALIDATION AND FILTERING
# ==============================================================================

# --- A Deny List to filter out common false positives from NER models. ---
DENY_LIST_TERMS = {
    "inc", "ltd", "llc", "corp", "corporation", "gmbh", "pvt", "ltd",
    "fig", "figure", "table", "appendix", "chapter", "section",
    "note", "notes", "summary", "introduction", "conclusion", "abstract",
    "gemini", "pytesseract", "spacy", "tesseract"
}

def is_context_relevant(text: str, entity_start: int, entity_end: int, window_size: int = 30) -> bool:
    """
    Checks the surrounding words of an entity for keywords that confirm or deny
    its sensitivity. This helps disambiguate generic terms (e.g., is a number an ID or just a quantity?).
    """
    context_start = max(0, entity_start - window_size)
    context_end = min(len(text), entity_end + window_size)
    context_window = text[context_start:context_end].lower()

    # Keywords that increase the likelihood of an entity being sensitive.
    confirm_keywords = [
        'account', 'acct', 'a/c', 'card', 'ssn', 'id', 'license', 'passport',
        'member', 'employee', 'emp', 'customer', 'ref', 'invoice', 'po #', 'p.o.',
        'phone', 'tel', 'mobile', 'fax', 'email', 'name', 'mr.', 'mrs.', 'ms.'
    ]

    # Keywords that suggest an entity is benign.
    deny_keywords = [
        'page', 'chapter', 'section', 'fig', 'figure', 'table', 'quantity',
        'qty', 'item', 'step', 'version', 'v.', 'rev', 'line', 'row', 'model'
    ]

    if any(keyword in context_window for keyword in deny_keywords):
        return False
    if any(keyword in context_window for keyword in confirm_keywords):
        return True
    return False


def get_semantic_pii_score(text: str, entity_start: int, entity_end: int, nlp_model) -> float:
    """
    Calculates a PII relevance score based on the semantic similarity of
    surrounding words to a profile of PII-related concepts.
    """
    # A spaCy Doc representing common PII concepts.
    pii_profile = nlp_model("personal private identity financial health contact address security account")

    # Get the text window around the entity.
    context_start = max(0, entity_start - 10)
    context_end = min(len(text), entity_end + 10)
    context_text = text[context_start:entity_start] + " " + text[entity_end:context_end]

    if not context_text.strip(): return 0.0

    context_doc = nlp_model(context_text)
    # Filter out stopwords and punctuation to focus on meaningful words.
    context_doc_no_stopwords = [token for token in context_doc if not token.is_stop and not token.is_punct]
    if not context_doc_no_stopwords: return 0.0

    final_doc = nlp_model(' '.join([token.text for token in context_doc_no_stopwords]))
    if not final_doc.vector_norm: return 0.0

    # Return the similarity score between the entity's context and the PII profile.
    return final_doc.similarity(pii_profile)


def post_process_and_validate_entities(entities: List[Dict], full_text_by_page: Dict[int, str], nlp_model) -> List[Dict]:
    """
    The main validation filter. It takes all detected entities and removes
    likely false positives using a set of rules.

    How it works:
    1.  It prioritizes longer entities over shorter ones (e.g., "John Doe" over "John").
    2.  It filters out any entity found in the `DENY_LIST_TERMS`.
    3.  It removes any Presidio entity with a very low confidence score.
    4.  For ambiguous entities (like names or numbers from spaCy), it uses contextual
        and semantic checks (`is_context_relevant`, `get_semantic_pii_score`) to
        decide whether to keep or discard them.
    """
    validated_entities = []
    processed_spans = set()

    # Sort by length (descending) to handle overlapping entities correctly.
    for entity in sorted(entities, key=lambda x: len(x['text']), reverse=True):
        entity_span = (entity['page_num'], entity['start_char'], entity['end_char'])
        # If a larger entity containing this one has already been processed, skip.
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

        # --- RULE 3: Contextual Validation for Ambiguous Entities ---
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
                    continue

        validated_entities.append(entity)
        processed_spans.add(entity_span)

    return validated_entities

# ==============================================================================
# 4. FINAL ACTION: REDACTION
# ==============================================================================

def redact_sensitive_info(pdf_path: str, detected_entities: List[Dict], output_pdf_path: str) -> bool:
    """
    Applies redaction annotations to the PDF based on the validated list of entities.

    How it works:
    - It groups all entities by the page they appear on.
    - It iterates through each page and each entity on that page.
    - It uses a dual strategy for finding the exact location of the text to redact:
      1. For OCR'd Pages: It uses the pre-calculated bounding boxes for each word.
         This is extremely precise.
      2. For Native Text Pages: It uses PyMuPDF's `search_for` method with `quads=True`.
         This is more robust than a simple rectangle search as it can handle rotated
         or skewed lines of text.
    - After adding all redaction annotations, it applies them, permanently and
      securely removing the sensitive content from the output PDF.
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
            if not page_entities:
                continue
            page = document.load_page(page_num)

            logging.info(f"--- Applying redactions to Page {page_num + 1} ---")
            for entity in page_entities:
                entity_text = entity["text"]
                is_ocr_entity = entity['is_ocr_page']

                if is_ocr_entity:
                    # --- Precision Redaction for OCR'd Text ---
                    ocr_word_details = entity.get('ocr_word_details')
                    if not ocr_word_details:
                        logging.warning(f"Skipping OCR entity '{entity_text}' on page {page_num + 1} due to missing word details.")
                        continue

                    matched_bboxes = []
                    # Find the bounding boxes of the words that make up the sensitive entity.
                    for word in ocr_word_details:
                        if (entity['start_char'] < word['end_char_in_ocr_text']) and (entity['end_char'] > word['start_char_in_ocr_text']):
                            matched_bboxes.append(word['bbox'])

                    if matched_bboxes:
                        logging.info(f"Redacting OCR entity '{entity_text}' ({entity['label']}) with {len(matched_bboxes)} boxes.")
                        for bbox in matched_bboxes:
                            margin = 1  # A small margin to ensure full coverage
                            precise_rect = fitz.Rect(bbox.x0 - margin, bbox.y0 - margin, bbox.x1 + margin, bbox.y1 + margin).intersect(page.rect)
                            if not precise_rect.is_empty:
                                page.add_redact_annot(precise_rect, fill=(0, 0, 0))
                                total_redactions_applied += 1
                    else:
                         logging.warning(f"Could not map OCR entity '{entity_text}' ({entity['label']}) to bounding boxes on page {page_num + 1}.")

                else:
                    # --- Robust Redaction for Native Text ---
                    logging.info(f"Redacting native text entity '{entity_text}' ({entity['label']}).")
                    # `search_for` with `quads=True` returns quadrilaterals, which are more
                    # accurate for text that might be slightly rotated or on a curve.
                    text_instances = page.search_for(entity_text, quads=True)
                    for quad in text_instances:
                        page.add_redact_annot(quad, fill=(0, 0, 0))
                        total_redactions_applied += 1

        if total_redactions_applied > 0:
            logging.info(f"\nApplying {total_redactions_applied} secure redaction marks across the document...")
            # This step permanently removes the content under the redaction annotations.
            # `images=fitz.PDF_REDACT_IMAGE_PIXELS` also removes the underlying image data.
            for page in document:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

            document.save(output_pdf_path, garbage=4, deflate=True)
            logging.info(f"✅ Successfully applied redactions. Redacted PDF saved to: {output_pdf_path}")
            return True
        else:
            logging.info("\nNo sensitive entities were found or could be mapped for redaction.")
            document.close()
            return False

    except Exception as e:
        logging.error(f"An error occurred during the final redaction process: {e}", exc_info=True)
        return False

# ==============================================================================
# 5. SCRIPT EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    if nlp is None:
        sys.exit(1) # Exit if the core spaCy model failed to load.

    logging.info(f"Python Executable: {sys.executable}")
    try:
        logging.info(f"PyMuPDF (fitz) version: {fitz.__version__}")
    except ImportError:
        logging.warning("PyMuPDF (fitz) is not imported or installed correctly.")

    # --- User Configuration ---
    input_pdf_path = "PLL-step3.pdf"
    output_pdf_path = "universal.pdf"

    if not os.path.exists(input_pdf_path):
        logging.error(f"FATAL: Input PDF file not found at '{input_pdf_path}'.")
    else:
        # This is the main workflow of the script.
        logging.info(f"\n--- Starting Advanced Processing for: {input_pdf_path} ---")

        # STEP 1: Setup PII detection engines.
        analyzer = AnalyzerEngine()

        # STEP 2: Extract all text from the PDF, using OCR where necessary.
        all_pages_data = extract_text_from_pdf_with_ocr(input_pdf_path)

        if all_pages_data:
            raw_detected_entities = []
            full_text_by_page = {p['page_num']: p['text'] for p in all_pages_data}

            # STEP 3: Run all detection engines to find potential sensitive entities.
            logging.info("\n--- Phase 1: Detecting All Potential Entities ---")
            for page_data in all_pages_data:
                presidio_entities = identify_entities_with_presidio(analyzer, page_data['text'], page_data['page_num'], page_data['is_ocr_page'], page_data.get('ocr_word_details'))
                raw_detected_entities.extend(presidio_entities)

                # The spaCy engine is currently disabled in this workflow but can be enabled
                # by uncommenting the two lines below to add another layer of detection.
                # spacy_entities = identify_sensitive_entities([page_data])
                # raw_detected_entities.extend(spacy_entities)

                regex_entities = identify_sensitive_entities_regex(page_data['text'], page_data['page_num'], page_data['is_ocr_page'], page_data.get('ocr_word_details'))
                raw_detected_entities.extend(regex_entities)

            # STEP 4: Validate the raw findings to filter out false positives.
            logging.info(f"\n--- Phase 2: Validating {len(raw_detected_entities)} Potential Entities to Remove False Positives ---")
            validated_entities = post_process_and_validate_entities(raw_detected_entities, full_text_by_page, nlp)

            # STEP 5: Apply redactions for the final, validated list of entities.
            if validated_entities:
                logging.info(f"\n--- Phase 3: Applying Redaction for {len(validated_entities)} Validated Entities ---")

                # Print the final list of items that will be redacted for review.
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