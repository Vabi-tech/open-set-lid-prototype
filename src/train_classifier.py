
import argparse
import json
import os
from typing import Dict, List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

from src.data_io import get_dataset


def build_hf(examples: List[Dict[str, str]], label_list: List[str], tokenizer):
    """Convert list[{"text","label"}] -> HF Dataset with tokenized inputs and class ids."""
    label2id = {lab: i for i, lab in enumerate(label_list)}
    ds = Dataset.from_list(examples)
    ds = ds.class_encode_column("label") if "label" in ds.column_names else ds

    def tok(batch):
        out = tokenizer(batch["text"], truncation=True, padding=False)
        out["labels"] = [label2id[str(lab)] for lab in batch["label"]]
        return out

    ds = ds.map(tok, batched=True, remove_columns=[c for c in ds.column_names if c not in ["text", "label"]])
    return ds, label2id


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tatoeba", "flores", "papluca"], default="papluca")
    parser.add_argument("--seen_langs", type=str, default="en,es,de,fr,it")
    parser.add_argument("--max_per_lang", type=int, default=2000)
    parser.add_argument("--model_name", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--output_dir", type=str, default="models/lid-seen")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    seen = [s.strip() for s in args.seen_langs.split(",") if s.strip()]
    data, label_list = get_dataset(args.dataset, seen, unseen_langs=[], max_per_lang=args.max_per_lang)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    train_ds, label2id = build_hf(data["train"], label_list, tok)
    val_ds, _ = build_hf(data["validation"], label_list, tok)
    test_ds, _ = build_hf(data["test"], label_list, tok)

    id2label = {v: k for k, v in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    # Version-tolerant TrainingArguments
    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_steps=50,
            report_to=[],  # no wandb
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            logging_steps=50,
        )

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    trainer.train()

    # Save everything locally
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # Evaluate on held-out test
    test_metrics = trainer.evaluate(test_ds)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
