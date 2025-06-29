# backend/services/entity_detection.py

import logging
import re
import sys
from typing import List, Dict, Any, Optional

import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
 # **[COPIED AS IS]** This was in your original redaction_core.py. It should ideally only be in main.py. Will address this later in main.py.

# **[ADDED]** This line initializes a logger specifically for this module.
logger = logging.getLogger(__name__)

# --- Tesseract Configuration ---
# **[COPIED AS IS]** This was in your original redaction_core.py, but is not directly used here.
# If tesseract.exe is not in your system's PATH, you might need to specify its path.
# For example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Load spaCy Language Model ---
# This model is used for advanced natural language processing tasks, specifically
# for Named Entity Recognition (NER) and for semantic analysis in the validation step.
 # **[COPIED AS IS]** This was in your original redaction_core.py. It should ideally only be in main.py. Will address this later in main.py.
try:
    # 1. Define the NLP engine configuration for Presidio.
    #    This explicitly tells Presidio to use the small spaCy model we installed.
    nlp_config = {
        # CRITICAL FIX: The key is 'nlp_engine_name', not 'nlp_engine'.
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
    }

    # 2. Create the NLP engine provider with the explicit configuration.
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()
    
    # Also load the spaCy model directly for any other potential uses.
    nlp = spacy.load("en_core_web_sm")

    # 3. Create the AnalyzerEngine using the correctly configured NLP engine.
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"]
    )
    
    logger.info("✅ Presidio Analyzer and spaCy model 'en_core_web_sm' loaded successfully.")

except Exception as e:
    logger.error(f"❌ A critical error occurred during NLP engine initialization: {e}")
    logger.error("The application may not be able to redact all entity types. Please check logs.")
    nlp = None
    analyzer = None
# nlp_engine = SpacyNlpEngine({"en": "en_core_web_sm"}) 

# Initialize Presidio Analyzer with the configured NLP engine
# This `analyzer` instance will now correctly use `en_core_web_sm`
# analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
# logger.info("Initialized Presidio Analyzer with en_core_web_sm.")
# ==============================================================================
# 2. SENSITIVE ENTITY DETECTION
# ==============================================================================

# def identify_sensitive_entities(pages_data: List[Dict]) -> List[Dict]:
#     """
#     Detects general named entities (PERSON, ORG, etc.) using the spaCy NER model.
#     """
#     if nlp is None:
#         logger.error("spaCy model not loaded, cannot identify entities.")
#         return []
#     all_sensitive_entities = []
#     # Defines the types of entities spaCy should look for.
#     PII_LABELS = ["PERSON", "NORP", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "MONEY", "TIME", "FAC"]
#     for page_data in pages_data:
#         doc = nlp(page_data['text'])
#         for ent in doc.ents:
#             if ent.label_ in PII_LABELS:
#                 all_sensitive_entities.append({"text": ent.text, "label": f"SPACY_{ent.label_}", "page_num": page_data['page_num'], "start_char": ent.start_char, "end_char": ent.end_char, "is_ocr_page": page_data['is_ocr_page'], "ocr_word_details": page_data.get('ocr_word_details', [])})
#     return all_sensitive_entities


def identify_sensitive_entities_regex(text: str, page_num: int, is_ocr_page: bool, ocr_word_details: List = None, allowed_types: Optional[List[str]] = None) -> List[Dict]:
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
        #"AMOUNT": r"\b(?:\d{1,3}(?:,\d{3})*|\d+)\b",
        "AMOUNT": r"\b(?:[1-9]\d{3,}|\d{1,3}(?:,\d{3})+)\b",
        #"LONG_NUMBERS": r"[₹]?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d{5,}(?:\.\d+)?\b",
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
        entity_label = f"REGEX_{label}"
        if allowed_types is None or entity_label in allowed_types:    
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


def identify_entities_with_presidio(analyzer: AnalyzerEngine, text: str, page_num: int, is_ocr_page: bool, ocr_word_details: List = None, allowed_types: Optional[List[str]] = None) -> List[Dict]:
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
        # ADDED filtering logic
       
        
        #results = analyzer.analyze(text=text, language='en', entities=presidio_labels_to_check)
        for res in results:
            entity_label = f"PRESIDIO_{res.entity_type}"
            # **CORRECTED LOGIC**: Filter *after* the analysis.
            if allowed_types is None or entity_label in allowed_types:
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
        logger.error(f"Presidio analysis failed on page {page_num + 1}: {e}")
    return presidio_entities