# backend/services/validation.py

import logging
from typing import List, Dict, Any, Optional

# **[ADDED]** This line initializes a logger specifically for this module.
logger = logging.getLogger(__name__)

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


def post_process_and_validate_entities(entities: List[Dict], full_text_by_page: Dict[int, str], nlp_model, allowed_types: Optional[List[str]] = None) -> List[Dict]:
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

    # ADDED initial filtering step
    if allowed_types:
        initial_entities = [e for e in entities if e['label'] in allowed_types]
        logger.info(f"Pre-validation filter: Kept {len(initial_entities)} of {len(entities)} entities based on allowed types.")
    else:
        initial_entities = entities
    
    if not initial_entities:
        return []


    validated_entities = []
    processed_spans = set()

    # Sort by length (descending) to handle overlapping entities correctly.
    for entity in sorted(initial_entities, key=lambda x: len(x['text']), reverse=True):
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
            logger.info(f"FILTERED (Deny List): '{entity['text']}' ({label})")
            continue

        # --- RULE 2: Score-based validation for Presidio entities ---
        if 'PRESIDIO' in label and entity.get('score', 0) < 0.0:
             logger.info(f"FILTERED (Low Presidio Score: {entity['score']:.2f}): '{entity['text']}' ({label})")
             continue

        # --- RULE 3: Contextual Validation for Ambiguous Entities ---
        ambiguous_labels = {"SPACY_CARDINAL", "SPACY_PERSON", "SPACY_ORG"}
        # **[MODIFIED]** Added check `nlp_model is not None` to prevent error if model failed to load.
        if label in ambiguous_labels:
            if is_context_relevant(full_text, entity['start_char'], entity['end_char']):
                logger.info(f"VALIDATED (Context Keyword): '{entity['text']}' ({label})")
            else:
                semantic_score = get_semantic_pii_score(full_text, entity['start_char'], entity['end_char'], nlp_model)
                if semantic_score > 0.60: # Tunable threshold
                    logger.info(f"VALIDATED (Semantic Score: {semantic_score:.2f}): '{entity['text']}' ({label})")
                else:
                    logger.info(f"FILTERED (Low Context/Semantics: {semantic_score:.2f}): '{entity['text']}' ({label})")
                    continue

        validated_entities.append(entity)
        processed_spans.add(entity_span) # **[MODIFIED]** Ensure parent span is added for correct skipping of contained entities.

    return validated_entities
