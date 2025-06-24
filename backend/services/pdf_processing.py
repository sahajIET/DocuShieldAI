# backend/services/pdf_processing.py

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
from PIL import Image # Explicitly import Image from PIL

# **[ADDED]** This line initializes a logger specifically for this module.
# **[ADDED]** This is a standard practice for Python modules to integrate with the main application's logging configuration.
logger = logging.getLogger(__name__)

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
    logger.info(f"Analysis: Contrast={contrast:.2f}, Noise={laplacian_var:.2f}. Selected Profile: '{selected_profile['description']}'")
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
                    logger.info(f"Page {page_num + 1}: Image-based page detected. Attempting OCR...")
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
                    logger.info(f"Page {page_num + 1}: Native text page detected.")
                    page_data['text'] = text_from_page

                all_pages_data.append(page_data)

            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1} in {pdf_path}. Error: {e}", exc_info=True)
                continue

        document.close()
        return all_pages_data
    except Exception as e:
        logger.error(f"Failed to open or read PDF {pdf_path}. Error: {e}", exc_info=True)
        return []