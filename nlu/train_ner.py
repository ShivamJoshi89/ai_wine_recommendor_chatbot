# nlu/train_ner.py
import os
import re
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUT_DIR = os.path.join(MODELS_DIR, "wine_ner_model")

entity_data_path = os.path.join(SCRIPT_DIR, "final_corrected_annotated_wine_queries.json")

entity_labels = [
    "O",
    "B-WINE_TYPE", "I-WINE_TYPE",
    "B-GRAPE_VARIETY", "I-GRAPE_VARIETY",
    "B-REGION", "I-REGION",
    "B-PRICE_RANGE", "I-PRICE_RANGE",
    "B-FOOD_PAIRING", "I-FOOD_PAIRING",
]
entity_label2id = {label: i for i, label in enumerate(entity_labels)}
entity_id2label = {i: label for label, i in entity_label2id.items()}

MODEL_CHECKPOINT = "bert-base-uncased"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_dataset("json", data_files=entity_data_path)
    if "test" not in ds.keys():
        ds = ds["train"].train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_offsets_mapping=True,
        )

        all_labels = []
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            label_ids = [-100] * len(tokenized["input_ids"][i])
            for ent in batch["entities"][i]:
                ent_start = ent["start"]
                ent_end = ent["end"]
                first = True
                for idx, (start, end) in enumerate(offsets):
                    if start == end:
                        continue
                    if end > ent_start and start < ent_end:
                        if first:
                            label_ids[idx] = entity_label2id["B-" + ent["label"]]
                            first = False
                        else:
                            label_ids[idx] = entity_label2id["I-" + ent["label"]]
            all_labels.append(label_ids)

        tokenized["labels"] = all_labels
        tokenized.pop("offset_mapping")
        return tokenized

    ds = ds.map(tokenize_and_align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(entity_labels),
        id2label=entity_id2label,
        label2id=entity_label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "_runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        # IMPORTANT: do NOT pass tokenizer=... (your transformers version rejects it)
    )

    print("Training NER model...")
    trainer.train()
    print("Evaluating...")
    print(trainer.evaluate())

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved NER model to: {OUT_DIR}")

if __name__ == "__main__":
    main()