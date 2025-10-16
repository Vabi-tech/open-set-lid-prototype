import argparse, os, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_io import get_dataset

@torch.no_grad()
def get_logits_and_features(model, feature_model, tokenizer, ds_list, label2id_map=None, batch_size=64, device="cpu"):
    def to_dl(texts, labels=None):
        enc = tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        if labels is None:
            return DataLoader(list(zip(enc["input_ids"], enc["attention_mask"])), batch_size=batch_size)
        return DataLoader(list(zip(enc["input_ids"], enc["attention_mask"], labels)), batch_size=batch_size)

    out = []
    for split_name, split in ds_list.items():
        texts = [x["text"] for x in split]
        labels = [x["label"] for x in split] if "label" in split[0] else None
        dl = to_dl(texts, labels)
        for batch in tqdm(dl, desc=split_name):
            if labels is None:
                input_ids, attn = batch
            else:
                input_ids, attn, y = batch
            input_ids = input_ids.to(device)
            attn = attn.to(device)

            logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, C]
            feats = feature_model(input_ids=input_ids, attention_mask=attn).last_hidden_state[:,0,:]  # CLS

            row = {
                "split": split_name,
                "logits": logits.cpu().numpy().tolist(),
                "features": feats.cpu().numpy().tolist()
            }
            if labels is not None:
                # y is a tuple of labels for this batch; usually strings like "en","fr",...
                y_list = list(y)
                if label2id_map is None:
                    # Fall back to model's config if not provided
                    label2id_map = getattr(model.config, "label2id", {})
                y_ids = [int(label2id_map.get(str(lbl), -1)) for lbl in y_list]
                row["label"] = y_ids
            out.append(row)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="tatoeba")
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--seen_langs", type=str, default="en,es,de,fr,it")
    ap.add_argument("--unseen_langs", type=str, default="pt,ru,sv")
    ap.add_argument("--max_per_lang", type=int, default=2000)
    ap.add_argument("--out_path", type=str, default="outputs/preds.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seen = [s.strip() for s in args.seen_langs.split(",") if s.strip()]
    unseen = [s.strip() for s in args.unseen_langs.split(",") if s.strip()]

    data, label_list = get_dataset(args.dataset, seen, unseen, max_per_lang=args.max_per_lang)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    clf = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()
    feat_model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()

    ds_list = {
        "seen_test": data["test"],
        "ood_test": data["ood_test"]
    }

    outputs = get_logits_and_features(clf, feat_model, tok, ds_list, label2id_map=getattr(clf.config, "label2id", {}), device=device)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        for row in outputs:
            f.write(json.dumps(row) + "\n")
    print("Wrote", args.out_path)

if __name__ == "__main__":
    main()