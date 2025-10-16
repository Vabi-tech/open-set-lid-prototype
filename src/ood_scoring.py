import numpy as np
from scipy.special import logsumexp


def softmax(x, axis=-1):
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex / ex.sum(axis=axis, keepdims=True)


def msp(logits):
    """Max Softmax Probability"""
    probs = softmax(logits, axis=-1)
    return probs.max(axis=-1)


def energy(logits, T=1.0):
    """Energy score: -T * logsumexp(logits / T) (lower = ID); we will use it as OOD score directly."""
    return -1.0 * logsumexp(logits / T, axis=-1)


def mahalanobis(features, class_means, precision):
    """Mahalanobis distance to nearest class mean (higher => more OOD-like)."""
    diffs = features[:, None, :] - class_means[None, :, :]
    m = np.einsum('ncd,dd,nce->nc', diffs, precision, diffs)
    return m.min(axis=1)


def class_stats(features, labels, num_classes):
    feats = np.asarray(features)
    y = np.asarray(labels)
    means = np.stack([feats[y == c].mean(axis=0)
                     for c in range(num_classes)], axis=0)
    centered = np.vstack([feats[y == c] - means[c]
                         for c in range(num_classes)])
    cov = np.cov(centered, rowvar=False) + 1e-5 * np.eye(centered.shape[1])
    precision = np.linalg.inv(cov)
    return means, precision
