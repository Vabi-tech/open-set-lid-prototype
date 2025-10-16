import numpy as np

def ece(probs, labels, n_bins=15):
    """Expected Calibration Error (ECE) with equal-width bins in [0,1]."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece_val = 0.0
    bin_confs, bin_accs, bin_counts = [], [], []

    for i in range(n_bins):
        l, r = bins[i], bins[i+1]
        idx = (confidences > l) & (confidences <= r)
        count = idx.sum()
        if count > 0:
            acc = (predictions[idx] == labels[idx]).mean()
            conf = confidences[idx].mean()
            ece_val += (count / len(labels)) * abs(acc - conf)
            bin_confs.append(conf)
            bin_accs.append(acc)
            bin_counts.append(int(count))
        else:
            bin_confs.append(0.0)
            bin_accs.append(0.0)
            bin_counts.append(0)
    return ece_val, (bins, np.array(bin_confs), np.array(bin_accs), np.array(bin_counts))