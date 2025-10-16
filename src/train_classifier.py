import argparse
import os
import json
from typing import List
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from src.data_io import get_dataset


def build_hf(ds_split, label_list):
    label2id = {l: i for i, l in enumerate(label_list)}

    def _map(example):
        return {"text": example["text"], "label": label2id[example["label"]]}
    ds = Dataset.from_list(list(map(_map, ds_split)))
    ds = ds.class_encode_column("label")
    return ds, label2id


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    import numpy as np
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="tatoeba",
        choices=[
            'tatoeba',
            'flores',
            'papluca'])
    ap.add_argument("--seen_langs", type=str, default="en,es,de,fr,it")
    ap.add_argument("--max_per_lang", type=int, default=5000)
    ap.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-multilingual-cased")
    ap.add_argument("--output_dir", type=str, default="models/lid-seen")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    seen = [s.strip() for s in args.seen_langs.split(",") if s.strip()]
    data, label_list = get_dataset(
        args.dataset, seen, unseen_langs=[], max_per_lang=args.max_per_lang)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128)

    train_ds, label2id = build_hf(data["train"], label_list)
    val_ds, _ = build_hf(data["validation"], label_list)
    test_ds, _ = build_hf(data["test"], label_list)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"])
    val_ds.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"])
    test_ds.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_list), id2label={
            i: l for i, l in enumerate(label_list)}, label2id=label2id)

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
            logging_steps=50,
            report_to=[]
        )
    except TypeError:
        # For older Transformers versions
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            logging_steps=50
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tok
    )
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = trainer.evaluate(test_ds)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Saved model to", args.output_dir)


if __name__ == "__main__":
    main()
