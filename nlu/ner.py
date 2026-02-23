# nlu/ner.py
import os
import re
from typing import Dict, List, Any, Optional

from transformers import pipeline

##################################
# PRICE FALLBACK HELPER
##################################
_price_pattern = re.compile(r"\$\s*\d+(?:\.\d+)?")

def fallback_price_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    If no PRICE_RANGE entity is detected, grab the first $<number> in the text.
    Returns a dict matching your entity schema or None.
    """
    m = _price_pattern.search(text)
    if not m:
        return None
    return {
        "label": "PRICE_RANGE",
        "word": m.group(),
        "score": 1.0,
        "start": m.start(),
        "end": m.end(),
    }

##################################
# PATHS (repo-root/models/...)
##################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")

INTENT_DIR = os.path.join(MODELS_DIR, "intent_model")
NER_DIR = os.path.join(MODELS_DIR, "wine_ner_model")

##################################
# POST-PROCESSING (keep it light)
##################################
def merge_adjacent_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not entities:
        return []
    merged = [entities[0]]
    for ent in entities[1:]:
        last = merged[-1]
        if ent["label"] == last["label"]:
            merged[-1]["word"] += " " + ent["word"]
            merged[-1]["end"] = ent.get("end", merged[-1].get("end"))
            merged[-1]["score"] = float(max(ent.get("score", 1.0), merged[-1].get("score", 1.0)))
        else:
            merged.append(ent)
    return merged

def advanced_filter_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep your business filtering logic. This is inference-only.
    """
    filtered: List[Dict[str, Any]] = []
    stopwords = {"a", "an", "the", "and", "with", "from", "of", "i", "im", "i'm"}
    punctuation_chars = " -–—,;:.!?'\""

    for ent in entities:
        word = str(ent.get("word", "")).strip(punctuation_chars).lower()
        label = ent.get("label", "")

        # Normalize pairing phrasing
        if label == "FOOD_PAIRING":
            word = word.replace("pairs well with", "").replace("well with", "").strip()

        # PRICE cleanup: keep only the $<number>
        if label == "PRICE_RANGE":
            m = re.search(r"\$\s*\d+(?:\.\d+)?", word)
            if not m or float(ent.get("score", 1.0)) < 0.5:
                continue
            word = m.group()

        # Basic guards
        if len(word) < 2:
            continue

        # Label-specific filters (same as your original intent, but safe)
        if label == "GRAPE_VARIETY":
            if word in stopwords or len(word) < 3:
                continue
            word = re.sub(r"\s+from\s*$", "", word).strip()
            ent["word"] = word
            if float(ent.get("score", 1.0)) < 0.7:
                continue

        ent["word"] = word
        ent["label"] = label
        filtered.append(ent)

    return filtered

def postprocess_pipeline_output(pipeline_output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for ent in pipeline_output:
        # HuggingFace aggregated NER output keys: entity_group, word, score, start, end
        converted.append(
            {
                "label": ent.get("entity_group") or ent.get("entity") or "O",
                "word": ent.get("word", ""),
                "score": float(ent.get("score", 1.0)),
                "start": int(ent.get("start", 0)),
                "end": int(ent.get("end", 0)),
            }
        )

    filtered = advanced_filter_entities(converted)
    merged = merge_adjacent_entities(filtered)
    return merged

##################################
# LAZY LOADERS (do NOT train here)
##################################
_intent_pipe = None
_ner_pipe = None

def _dir_has_model(model_dir: str) -> bool:
    return os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json"))

def _load_intent_pipeline():
    global _intent_pipe
    if _intent_pipe is not None:
        return _intent_pipe

    if _dir_has_model(INTENT_DIR):
        _intent_pipe = pipeline("text-classification", model=INTENT_DIR, tokenizer=INTENT_DIR)
    else:
        _intent_pipe = None
    return _intent_pipe

def _load_ner_pipeline():
    global _ner_pipe
    if _ner_pipe is not None:
        return _ner_pipe

    if _dir_has_model(NER_DIR):
        _ner_pipe = pipeline("ner", model=NER_DIR, tokenizer=NER_DIR, aggregation_strategy="simple")
    else:
        _ner_pipe = None
    return _ner_pipe

##################################
# PUBLIC API (used by backend)
##################################
def classify_intent(text: str) -> str:
    """
    Returns an intent label. If no trained model is present, returns a safe default.
    """
    pipe = _load_intent_pipeline()
    if pipe is None:
        return "recommend_wine"
    result = pipe(text)
    return result[0].get("label", "recommend_wine")

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Returns a list of extracted entities. If no trained NER model is present,
    falls back to only PRICE extraction.
    """
    pipe = _load_ner_pipeline()
    if pipe is None:
        ents: List[Dict[str, Any]] = []
        p = fallback_price_from_text(text)
        if p:
            ents.append(p)
        return ents

    raw = pipe(text)
    return postprocess_pipeline_output(raw)

def process_query(query: str) -> Dict[str, Any]:
    intent = classify_intent(query)
    entities = extract_entities(query)

    # PRICE fallback if none found
    found_price = any(ent.get("label") == "PRICE_RANGE" for ent in entities)
    if not found_price:
        price_ent = fallback_price_from_text(query)
        if price_ent:
            entities.append(price_ent)

    return {
        "intent": intent,
        "entities_pipeline": entities,
    }