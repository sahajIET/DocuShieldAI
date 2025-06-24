# backend/services/pdf_redaction.py

import logging
from typing import List, Dict, Any

import fitz

# **[ADDED]** This line initializes a logger specifically for this module.
logger = logging.getLogger(__name__)

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
      1. For OCR'd Text: A highly accurate word-by-word bounding box mapping.
      2. For Native PDF Text: A robust search using text quads for accurate placement.
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

            logger.info(f"--- Applying redactions to Page {page_num + 1} ---")
            for entity in page_entities:
                entity_text = entity["text"]
                is_ocr_entity = entity['is_ocr_page']

                if is_ocr_entity:
                    # --- Precision Redaction for OCR'd Text ---
                    ocr_word_details = entity.get('ocr_word_details')
                    if not ocr_word_details:
                        logger.warning(f"Skipping OCR entity '{entity_text}' on page {page_num + 1} due to missing word details.")
                        continue

                    matched_bboxes = []
                    # Find the bounding boxes of the words that make up the sensitive entity.
                    for word in ocr_word_details:
                        if (entity['start_char'] < word['end_char_in_ocr_text']) and (entity['end_char'] > word['start_char_in_ocr_text']):
                            matched_bboxes.append(word['bbox'])

                    if matched_bboxes:
                        logger.info(f"Redacting OCR entity '{entity_text}' ({entity['label']}) with {len(matched_bboxes)} boxes.")
                        for bbox in matched_bboxes:
                            margin = 1  # A small margin to ensure full coverage
                            precise_rect = fitz.Rect(bbox.x0 - margin, bbox.y0 - margin, bbox.x1 + margin, bbox.y1 + margin).intersect(page.rect)
                            if not precise_rect.is_empty:
                                page.add_redact_annot(precise_rect, fill=(0, 0, 0))
                                total_redactions_applied += 1
                    else:
                         logger.warning(f"Could not map OCR entity '{entity_text}' ({entity['label']}) to bounding boxes on page {page_num + 1}.")

                else:
                    # --- Robust Redaction for Native Text ---
                    logger.info(f"Redacting native text entity '{entity_text}' ({entity['label']}).")
                    # `search_for` with `quads=True` returns quadrilaterals, which are more
                    # accurate for text that might be slightly rotated or on a curve.
                    text_instances = page.search_for(entity_text, quads=True)
                    for quad in text_instances:
                        page.add_redact_annot(quad, fill=(0, 0, 0))
                        total_redactions_applied += 1

        if total_redactions_applied > 0:
            logger.info(f"\nApplying {total_redactions_applied} secure redaction marks across the document...")
            # This step permanently removes the content under the redaction annotations.
            # `images=fitz.PDF_REDACT_IMAGE_PIXELS` also removes the underlying image data.
            for page in document:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

            document.save(output_pdf_path, garbage=4, deflate=True)
            logger.info(f"✅ Successfully applied redactions. Redacted PDF saved to: {output_pdf_path}")
            return True
        else:
            logger.info("\nNo sensitive entities were found or could be mapped for redaction.")
            document.close()
            return False

    except Exception as e:
        logger.error(f"An error occurred during the final redaction process: {e}", exc_info=True)
        return False