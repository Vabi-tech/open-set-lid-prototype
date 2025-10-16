import argparse, os, json, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from src.ood_scoring import msp, energy, class_stats, mahalanobis, softmax
from src.calibration import ece

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def assemble_arrays(rows, num_classes):
    data = {"seen_logits": [], "seen_feats": [], "seen_labels": [],
            "ood_logits": [], "ood_feats": []}
    for r in rows:
        logits = np.array(r["logits"])
        feats = np.array(r["features"])
        if r["split"] == "seen_test":
            labels = np.array(r["label"])
            data["seen_logits"].append(logits)
            data["seen_feats"].append(feats)
            data["seen_labels"].append(labels)
        elif r["split"] == "ood_test":
            data["ood_logits"].append(logits)
            data["ood_feats"].append(feats)
    for k in data:
        if len(data[k])>0:
            data[k] = np.concatenate(data[k], axis=0)
    return data

def pr_plot(y_true, score, title, out_path):
    precision, recall, _ = precision_recall_curve(y_true, score)
    ap = average_precision_score(y_true, score)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return ap

def reliability_plot(probs, labels, out_path):
    e, (bins, confs, accs, counts) = ece(probs, labels, n_bins=15)
    centers = 0.5*(bins[:-1]+bins[1:])
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={e:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    rows = load_jsonl(args.preds)
    first_seen = next(r for r in rows if r["split"]=="seen_test")
    num_classes = args.num_classes or len(first_seen["logits"][0])

    data = assemble_arrays(rows, num_classes)
    os.makedirs(args.out_dir, exist_ok=True)

    id_logits = data["seen_logits"]
    ood_logits = data["ood_logits"]
    id_feats = data["seen_feats"]
    ood_feats = data["ood_feats"]
    y = np.concatenate([np.zeros(len(id_logits)), np.ones(len(ood_logits))])

    msp_id = msp(id_logits); msp_ood = msp(ood_logits)
    msp_score = np.concatenate([1.0 - msp_id, 1.0 - msp_ood])

    en_id = energy(id_logits); en_ood = energy(ood_logits)
    en_score = np.concatenate([en_id, en_ood])

    id_labels = data["seen_labels"]
    means, precision = class_stats(id_feats, id_labels, num_classes)
    maha_id = mahalanobis(id_feats, means, precision)
    maha_ood = mahalanobis(ood_feats, means, precision)
    maha_score = np.concatenate([maha_id, maha_ood])

    ap_msp = pr_plot(y, msp_score, "PR — OOD (1 - MSP)", os.path.join(args.out_dir, "pr_msp.png"))
    ap_en  = pr_plot(y, en_score,  "PR — OOD (Energy)", os.path.join(args.out_dir, "pr_energy.png"))
    ap_ma  = pr_plot(y, maha_score,"PR — OOD (Mahalanobis)", os.path.join(args.out_dir, "pr_mahalanobis.png"))

    from src.ood_scoring import softmax
    probs_id = softmax(id_logits, axis=-1)
    ece_val = reliability_plot(probs_id, id_labels, os.path.join(args.out_dir, "reliability.png"))

    metrics = {
        "average_precision": {"ood_1_minus_msp": float(ap_msp), "ood_energy": float(ap_en), "ood_mahalanobis": float(ap_ma)},
        "ece_on_seen_test": float(ece_val)
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    print("Saved metrics and plots to", args.out_dir)

if __name__ == "__main__":
    main()