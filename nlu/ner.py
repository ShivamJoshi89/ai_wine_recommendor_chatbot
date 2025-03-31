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

##################################
# CONFIGURATION & PATHS
##################################

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Since this script is in the nlu folder and your JSON files are also in this folder,
# we join the file names directly with script_dir.
intent_data_path = os.path.join(script_dir, "sample_intent_data.json")
entity_data_path = os.path.join(script_dir, "final_corrected_annotated_wine_queries.json")

# ----- Intent Labels -----
intent_labels = ["recommend_wine", "food_pairing", "product_details"]
intent_label2id = {label: idx for idx, label in enumerate(intent_labels)}
intent_id2label = {idx: label for label, idx in intent_label2id.items()}

# ----- Entity Labels (BIO scheme) -----
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

# Model checkpoints
intent_model_checkpoint = "distilbert-base-uncased"
entity_model_checkpoint = "bert-base-uncased"

##################################
# INTENT CLASSIFICATION TRAINING
##################################

# Load synthetic intent data (replace with your production data later)
with open(intent_data_path, "r") as f:
    intent_data = json.load(f)
intent_dataset = Dataset.from_dict(intent_data)
intent_dataset = intent_dataset.train_test_split(test_size=0.2)

# Tokenizer for intent classification.
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_checkpoint)

def tokenize_intent(examples):
    return intent_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

tokenized_intent_dataset = intent_dataset.map(tokenize_intent, batched=True)

def add_intent_label_ids(examples):
    examples["label"] = [intent_label2id[intent] for intent in examples["intent"]]
    return examples

tokenized_intent_dataset = tokenized_intent_dataset.map(add_intent_label_ids, batched=True)

# Load the DistilBERT model for intent classification.
intent_model = AutoModelForSequenceClassification.from_pretrained(
    intent_model_checkpoint,
    num_labels=len(intent_labels),
    id2label=intent_id2label,
    label2id=intent_label2id
)

training_args_intent = TrainingArguments(
    output_dir="./intent_model",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10
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

intent_model.save_pretrained("./intent_model")
intent_tokenizer.save_pretrained("./intent_model")

# Create an intent classification pipeline.
intent_classifier = pipeline(
    "text-classification",
    model="./intent_model",
    tokenizer="./intent_model"
)

def classify_intent(text):
    result = intent_classifier(text)
    return result[0]['label']

##################################
# ENTITY EXTRACTION (NER) TRAINING
##################################

# Load entity annotation data.
entity_dataset = load_dataset("json", data_files=entity_data_path)
if "test" not in entity_dataset.keys():
    entity_dataset = entity_dataset["train"].train_test_split(test_size=0.2)

# Tokenizer for entity extraction.
entity_tokenizer = AutoTokenizer.from_pretrained(entity_model_checkpoint, use_fast=True)

def tokenize_and_align_labels_entities(examples):
    tokenized_inputs = entity_tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True
    )
    all_labels = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        label_ids = [-100] * len(tokenized_inputs["input_ids"][i])
        entities = examples["entities"][i]
        for entity in entities:
            ent_start = entity["start"]
            ent_end = entity["end"]
            ent_label = entity["label"]
            first_token = True
            for idx, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                if end > ent_start and start < ent_end:
                    if first_token:
                        label_ids[idx] = entity_label2id["B-" + ent_label]
                        first_token = False
                    else:
                        label_ids[idx] = entity_label2id["I-" + ent_label]
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

training_args_entity = TrainingArguments(
    output_dir="./wine_ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2
)

data_collator_entity = DataCollatorForTokenClassification(entity_tokenizer)

trainer_entity = Trainer(
    model=entity_model,
    args=training_args_entity,
    train_dataset=tokenized_entity_datasets["train"],
    eval_dataset=tokenized_entity_datasets["test"],
    tokenizer=entity_tokenizer,
    data_collator=data_collator_entity
)

print("Training entity extraction model...")
trainer_entity.train()
entity_eval = trainer_entity.evaluate()
print("Entity Evaluation:", entity_eval)

entity_model.save_pretrained("./wine_ner_model")
entity_tokenizer.save_pretrained("./wine_ner_model")

# Create a pipeline for entity extraction.
entity_extraction_pipeline = pipeline(
    "ner",
    model="./wine_ner_model",
    tokenizer="./wine_ner_model",
    aggregation_strategy="simple"
)

##################################
# POST-PROCESSING FUNCTIONS FOR NER
##################################

def predict_entities(text):
    entity_model.eval()
    inputs = entity_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(entity_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = entity_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    pred_label_ids = torch.argmax(outputs.logits, dim=-1)
    token_ids = inputs["input_ids"][0]
    tokens = entity_tokenizer.convert_ids_to_tokens(token_ids)
    predicted_labels = [entity_id2label[label_id.item()] for label_id in pred_label_ids[0]]
    
    entities = []
    current_entity = None
    for token, label in zip(tokens, predicted_labels):
        if token in entity_tokenizer.all_special_tokens:
            continue
        # Skip tokens that are punctuation or too short
        if len(token.strip("##")) <= 1 or re.fullmatch(r'[\W_]+', token):
            continue
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"label": label[2:], "word": token}
        elif label.startswith("I-") and current_entity is not None:
            if token.startswith("##"):
                token_clean = token.replace("##", "")
                current_entity["word"] += token_clean  # join without space
            else:
                current_entity["word"] += " " + token
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities

def merge_adjacent_entities(entities):
    """
    Merge adjacent entities with the same label.
    """
    if not entities:
        return []
    merged = [entities[0]]
    for ent in entities[1:]:
        last = merged[-1]
        if ent["label"] == last["label"]:
            merged[-1]["word"] += " " + ent["word"]
        else:
            merged.append(ent)
    return merged

def advanced_filter_entities(entities):
    """
    Applies domain-specific filtering:
      - For PRICE_RANGE, the entity must match a price pattern and have minimum confidence.
      - For FOOD_PAIRING, enforce a minimum confidence.
      - For GRAPE_VARIETY, drop stopwords and low-confidence predictions.
      - For WINE_TYPE and REGION, use valid domain lists.
      - Remove extra punctuation from the beginning and end of words.
    """
    filtered = []
    stopwords = {"a", "an", "the", "and", "with", "from", "of", "i", "im", "i'm"}
    punctuation_chars = " -–—,;:.!?'\""
    for ent in entities:
        word = ent["word"].strip(punctuation_chars).lower()
        # Clean food pairing phrases
        if ent["label"] == "FOOD_PAIRING":
            word = word.replace("pairs well with", "").replace("well with", "").strip()
        ent["word"] = word
        if len(word) < 2:
            continue
        if ent["label"] == "PRICE_RANGE":
            if not re.search(r'(under\s+)?\$\s*\d+', word):
                continue
            if ent.get("score", 1.0) < 0.5:
                continue
        if ent["label"] == "FOOD_PAIRING" and ent.get("score", 1.0) < 0.5:
            continue
        if ent["label"] == "WINE_TYPE":
            valid_wine_types = {"red wine", "white wine", "sparkling wine", "rosé wine", "fortified wine"}
            if word not in valid_wine_types:
                continue
        if ent["label"] == "REGION":
            valid_regions = {"italy", "france", "united states", "napa valley", "champagne", "barolo"}
            if word not in valid_regions:
                continue
        if ent["label"] == "GRAPE_VARIETY":
            if word in stopwords or len(word) < 3:
                continue
            word = re.sub(r'\s+from\s*$', '', word)
            ent["word"] = word
            if ent.get("score", 1.0) < 0.7:
                continue
        filtered.append(ent)
    return filtered

def postprocess_pipeline_output(pipeline_output):
    """
    Converts the Hugging Face pipeline output to our internal format,
    applies advanced filtering, and merges adjacent entities.
    """
    converted = []
    for ent in pipeline_output:
        converted.append({
            "label": ent["entity_group"],
            "word": ent["word"],
            "score": ent["score"],
            "start": ent["start"],
            "end": ent["end"]
        })
    filtered = advanced_filter_entities(converted)
    merged = merge_adjacent_entities(filtered)
    return merged

##################################
# UTILITY: LOAD FULL NER PIPELINE (BATCH VERSION)
##################################

def load_full_ner_pipeline(model_dir="./wine_ner_model"):
    """
    Loads the saved NER pipeline and wraps it so that every query is automatically postprocessed.
    """
    ner_pipe = pipeline(
        "ner",
        model=model_dir,
        tokenizer=model_dir,
        aggregation_strategy="simple"
    )
    def full_inference(query):
        raw_output = ner_pipe(query)
        processed_output = postprocess_pipeline_output(raw_output)
        return processed_output
    return full_inference

##################################
# COMBINED NLU INFERENCE (BATCH)
##################################

def process_query(query):
    """
    Processes an incoming query by:
      1. Classifying its intent.
      2. Extracting entities using both our custom inference and pipeline.
    Returns a dictionary with the intent and entities.
    """
    # Intent classification
    intent_result = intent_classifier(query)
    intent = intent_result[0]['label']
    
    # Entity extraction using our custom function
    entities_custom = predict_entities(query)
    entities_custom = advanced_filter_entities(entities_custom)
    entities_custom = merge_adjacent_entities(entities_custom)
    
    # Entity extraction using pipeline and postprocessing
    pipeline_output = entity_extraction_pipeline(query)
    entities_pipeline = postprocess_pipeline_output(pipeline_output)
    
    return {
        "intent": intent,
        "entities_custom": entities_custom,
        "entities_pipeline": entities_pipeline
    }

##################################
# PROCESSING A BATCH OF QUERIES VIA DATASET
##################################

# Define 15 sample queries
sample_queries = [
    "Looking for an affordable Pinot Noir wine.",
    "Suggest a white wine made from Chenin Blanc from France.",
    "Can you recommend a red wine that pairs well with steak?",
    "Can you recommend a fortified wine that pairs well with seafood?",
    "What wine pairs well with seafood?",
    "What wine pairs well with poultry?",
    "Can you recommend a red wine that pairs well with salad?",
    "What wine pairs well with steak?",
    "I need a premium wine from Oakville.",
    "What is a good Cabernet Sauvignon wine under $162?",
    "Find me a premium red wine from Dry Creek Valley.",
    "Suggest a white wine made from Chardonnay from France.",
    "Can you recommend a sparkling wine that pairs well with cheese?",
    "What wine pairs well with dessert?",
    "Can you recommend a red wine that pairs well with cheese?"
]

# Create a Dataset from the queries
query_dataset = Dataset.from_dict({"query": sample_queries})

# Map our process_query function over the dataset (batch mode for efficiency)
def process_query_batch(batch):
    outputs = [process_query(q) for q in batch["query"]]
    return {"nlu": outputs}

processed_dataset = query_dataset.map(process_query_batch, batched=True)

# Convert results to a dictionary and print with JSON formatting
results_dict = {f"Query {i+1}": {"query": q, "nlu": out} 
                for i, (q, out) in enumerate(zip(query_dataset["query"], processed_dataset["nlu"]))}

print(json.dumps(results_dict, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x))

# For demonstration, load the full NER pipeline and process the first query
full_ner = load_full_ner_pipeline()
processed_entities = full_ner(sample_queries[0])
print("\nProcessed NER Output from loaded pipeline for Query 1:")
print(json.dumps(processed_entities, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x))