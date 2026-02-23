# nlu/train_intent.py
import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUT_DIR = os.path.join(MODELS_DIR, "intent_model")

intent_data_path = os.path.join(SCRIPT_DIR, "sample_intent_data.json")

intent_labels = ["recommend_wine", "food_pairing", "product_details"]
intent_label2id = {label: idx for idx, label in enumerate(intent_labels)}
intent_id2label = {idx: label for label, idx in intent_label2id.items()}

MODEL_CHECKPOINT = "distilbert-base-uncased"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(intent_data_path, "r", encoding="utf-8") as f:
        intent_data = json.load(f)

    ds = Dataset.from_dict(intent_data).train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    ds = ds.map(tokenize, batched=True)

    def add_labels(batch):
        batch["label"] = [intent_label2id[i] for i in batch["intent"]]
        return batch

    ds = ds.map(add_labels, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(intent_labels),
        id2label=intent_id2label,
        label2id=intent_label2id,
    )

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "_runs"),
        eval_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        # IMPORTANT: do NOT pass tokenizer=... (your transformers version rejects it)
    )

    print("Training intent classifier...")
    trainer.train()
    print("Evaluating...")
    print(trainer.evaluate())

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved intent model to: {OUT_DIR}")

if __name__ == "__main__":
    main()