# enhanced_ner.py

import os
import re
import json
import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

############################################
# PRICE FALLBACK HELPER
############################################
_price_pattern = re.compile(r'\$\s*\d+(?:\.\d+)?')

def fallback_price_from_text(text):
    """
    If no PRICE_RANGE entity is detected, search for something like "$<number>"
    and return a dict or None if not found.
    """
    m = _price_pattern.search(text)
    if not m:
        return None
    return {
        "label": "PRICE_RANGE",
        "word": m.group(),
        "score": 1.0,
        "start": m.start(),
        "end": m.end()
    }

############################################
# CONFIGURATION & PATHS
############################################
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to your JSON data
intent_data_path = os.path.join(script_dir, "sample_intent_data.json")
entity_data_path = os.path.join(script_dir, "final_corrected_annotated_wine_queries.json")

# Intent labels
intent_labels = ["recommend_wine", "food_pairing", "product_details"]
intent_label2id = {label: idx for idx, label in enumerate(intent_labels)}
intent_id2label = {idx: label for label, idx in intent_label2id.items()}

# Entity labels (BIO scheme)
entity_labels = [
    "O",
    "B-WINE_TYPE", "I-WINE_TYPE",
    "B-GRAPE_VARIETY", "I-GRAPE_VARIETY",
    "B-REGION", "I-REGION",
    "B-PRICE_RANGE", "I-PRICE_RANGE",
    "B-FOOD_PAIRING", "I-FOOD_PAIRING"
]
entity_label2id = {label: i for i, label in enumerate(entity_labels)}
entity_id2label = {i: label for label, i in entity_label2id.items()}

# Pretrained model checkpoints
intent_model_checkpoint = "distilbert-base-uncased"
entity_model_checkpoint = "bert-base-uncased"

############################################
# INTENT CLASSIFICATION TRAINING
############################################
with open(intent_data_path, "r") as f:
    intent_data = json.load(f)

# Convert to HF Dataset
intent_dataset = Dataset.from_dict(intent_data)
intent_dataset = intent_dataset.train_test_split(test_size=0.2)

intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_checkpoint)

def tokenize_intent(examples):
    return intent_tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=64
    )

tokenized_intent_dataset = intent_dataset.map(tokenize_intent, batched=True)

def add_intent_label_ids(examples):
    examples["label"] = [intent_label2id[intent] for intent in examples["intent"]]
    return examples

tokenized_intent_dataset = tokenized_intent_dataset.map(add_intent_label_ids, batched=True)

intent_model = AutoModelForSequenceClassification.from_pretrained(
    intent_model_checkpoint,
    num_labels=len(intent_labels),
    id2label=intent_id2label,
    label2id=intent_label2id
)

training_args_intent = TrainingArguments(
    output_dir="./intent_model",
    evaluation_strategy="epoch",
    num_train_epochs=5,  # up from 3
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no"
)

trainer_intent = Trainer(
    model=intent_model,
    args=training_args_intent,
    train_dataset=tokenized_intent_dataset["train"],
    eval_dataset=tokenized_intent_dataset["test"],
    tokenizer=intent_tokenizer
)

print("Training intent classifier...")
trainer_intent.train()
intent_eval = trainer_intent.evaluate()
print("Intent Evaluation:", intent_eval)

# Save the trained intent model
intent_model.save_pretrained("./intent_model")
intent_tokenizer.save_pretrained("./intent_model")

# Pipeline for easy intent inference
intent_classifier = pipeline(
    "text-classification",
    model="./intent_model",
    tokenizer="./intent_model"
)

def classify_intent(text):
    result = intent_classifier(text)
    return result[0]['label']

############################################
# ENTITY EXTRACTION (NER) TRAINING
############################################
entity_dataset = load_dataset("json", data_files=entity_data_path)
if "test" not in entity_dataset.keys():
    entity_dataset = entity_dataset["train"].train_test_split(test_size=0.2)

entity_tokenizer = AutoTokenizer.from_pretrained(entity_model_checkpoint, use_fast=True)

def tokenize_and_align_labels_entities(examples):
    """
    Convert text to tokens, then align BIO tags using offset mapping.
    """
    tokenized_inputs = entity_tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True
    )

    all_labels = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        label_ids = [-100] * len(offsets)
        for ent in examples["entities"][i]:
            ent_start = ent["start"]
            ent_end = ent["end"]
            label_str = "B-" + ent["label"]  # start with B-
            first_token = True

            for idx, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                # if there's overlap
                if end > ent_start and start < ent_end:
                    if first_token:
                        label_ids[idx] = entity_label2id.get(label_str, 0)
                        first_token = False
                        label_str = "I-" + ent["label"]  # subsequent
                    else:
                        label_ids[idx] = entity_label2id.get(label_str, 0)
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

tokenized_entity_datasets = entity_dataset.map(tokenize_and_align_labels_entities, batched=True)

entity_model = AutoModelForTokenClassification.from_pretrained(
    entity_model_checkpoint,
    num_labels=len(entity_labels),
    id2label=entity_id2label,
    label2id=entity_label2id
)

# We'll compute token-level precision/recall/f1 with seqeval
from seqeval.metrics import precision_score, recall_score, f1_score

def compute_metrics_ner(p):
    pred_logits = p.predictions
    labels = p.label_ids
    pred_ids = np.argmax(pred_logits, axis=-1)

    true_entities = []
    pred_entities = []

    for pred_seq, gold_seq in zip(pred_ids, labels):
        t_tags = []
        p_tags = []
        for (pred_id, gold_id) in zip(pred_seq, gold_seq):
            if gold_id == -100:
                continue
            t_tags.append(entity_id2label[gold_id])
            p_tags.append(entity_id2label[pred_id])
        true_entities.append(t_tags)
        pred_entities.append(p_tags)

    precision = precision_score(true_entities, pred_entities)
    recall = recall_score(true_entities, pred_entities)
    f1 = f1_score(true_entities, pred_entities)
    return {"precision": precision, "recall": recall, "f1": f1}

data_collator_entity = DataCollatorForTokenClassification(entity_tokenizer)

training_args_entity = TrainingArguments(
    output_dir="./wine_ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,  # up from 5
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=2
)

trainer_entity = Trainer(
    model=entity_model,
    args=training_args_entity,
    train_dataset=tokenized_entity_datasets["train"],
    eval_dataset=tokenized_entity_datasets["test"],
    tokenizer=entity_tokenizer,
    data_collator=data_collator_entity,
    compute_metrics=compute_metrics_ner
)

print("Training entity extraction model...")
trainer_entity.train()
entity_eval = trainer_entity.evaluate()
print("Entity Evaluation:", entity_eval)

entity_model.save_pretrained("./wine_ner_model")
entity_tokenizer.save_pretrained("./wine_ner_model")

# Quick pipeline for NER inference
entity_extraction_pipeline = pipeline(
    "ner",
    model="./wine_ner_model",
    tokenizer="./wine_ner_model",
    aggregation_strategy="simple"
)

############################################
# ADVANCED FILTERING LOGIC
############################################

MIN_SCORE_THRESHOLD = 0.80  # Discard predictions below this confidence

# Regex for valid price references: "under $30", "$50", "above 100 dollars", etc.
price_regex = re.compile(
    r"(?:under|above|below)\s*\$?\d+|\$\s*\d+|\d+\s*dollars\b",
    flags=re.IGNORECASE
)

# Example domain dictionaries
valid_wine_types = {
    "red wine", "white wine", "sparkling wine", "rosé wine",
    "ros wine", "fortified wine", "dessert wine"
}
valid_regions = {
    "napa valley", "france", "champagne", "barossa valley", "rioja",
    "tuscany", "italy", "oregon", "washington", "california", "bordeaux",
    "burgundy", "maipo valley"  # etc...
}
valid_food = {
    "steak", "seafood", "poultry", "spicy food", "cheese", "salad", 
    "dessert", "pizza", "pasta", "chocolate", "charcuterie"
}

def advanced_filter_entities(entities):
    """
    Filter out nonsense or spurious predictions:
    1) Score threshold
    2) Label-specific checks (regex for PRICE_RANGE, dictionary checks, etc.)
    """
    filtered = []
    for ent in entities:
        label = ent["label"]
        word = ent["word"].lower().strip(".,!?-\"'()[]")  # basic cleanup
        score = ent["score"]

        # 1) Score threshold
        if score < MIN_SCORE_THRESHOLD:
            continue

        # 2) Label-specific logic
        if label == "PRICE_RANGE":
            if not price_regex.search(word):
                # if "affordable" or "for" got labeled incorrectly, skip
                continue
        elif label == "WINE_TYPE":
            if word not in valid_wine_types:
                continue
        elif label == "REGION":
            if word not in valid_regions:
                continue
        elif label == "FOOD_PAIRING":
            if word not in valid_food:
                continue
        elif label == "GRAPE_VARIETY":
            # optional: skip single-letter, or check known grapes, etc.
            # e.g. if len(word) < 3: continue
            pass

        # If it survived, keep it
        filtered.append(ent)
    return filtered

############################################
# POSTPROCESS: Convert pipeline output & filter
############################################
def postprocess_pipeline_output(pipeline_output):
    """
    Convert pipeline output format into a simpler list of:
    { label, word, score, start, end }
    Then apply advanced filtering.
    """
    # raw pipeline output has "entity_group", "word", "score", "start", "end"
    results = []
    for ent in pipeline_output:
        results.append({
            "label": ent["entity_group"],
            "word": ent["word"],
            "score": ent["score"],
            "start": ent["start"],
            "end": ent["end"]
        })

    # Filter out spurious predictions
    final = advanced_filter_entities(results)
    return final

############################################
# MAIN INFERENCE FUNCTION
############################################
def process_query(query):
    """
    - Classify intent
    - Extract NER pipeline results
    - Postprocess (filter) them
    - If no PRICE_RANGE is found, fallback to searching for $<number> in text
    """
    # 1) Intent
    intent = classify_intent(query)

    # 2) Raw pipeline output
    raw_ents = entity_extraction_pipeline(query)
    # 3) Postprocess
    entities = postprocess_pipeline_output(raw_ents)

    # 4) Price fallback
    found_price = any(ent["label"] == "PRICE_RANGE" for ent in entities)
    if not found_price:
        fallback_ent = fallback_price_from_text(query)
        if fallback_ent:
            entities.append(fallback_ent)

    return {
        "intent": intent,
        "entities": entities
    }

############################################
# BATCH PROCESS EXAMPLE
############################################
if __name__ == "__main__":
    sample_queries = [
        "Looking for an affordable Pinot Noir wine.",
        "Suggest a white wine made from Chenin Blanc from France.",
        "Can you recommend a red wine that pairs well with steak?",
        "I want a red wine under $30 for spicy food.",
        "recommend me a red wine from napa valley which is above 150 dollars",
        # Some additional tricky ones:
        "Can I get a white wine for dessert from oregon that is above $25?",
        "I need a cheap syrah from barossa valley"
    ]

    from datasets import Dataset
    query_dataset = Dataset.from_dict({"query": sample_queries})

    def process_query_batch(batch):
        return {"nlu": [process_query(q) for q in batch["query"]]}

    processed = query_dataset.map(process_query_batch, batched=True)
    print(json.dumps(processed["nlu"], indent=2))
