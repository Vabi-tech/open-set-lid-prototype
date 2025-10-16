import random
from typing import List
from datasets import load_dataset

LANG_ALIAS = {
    "en": "eng",
    "es": "spa",
    "de": "deu",
    "fr": "fra",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "sv": "swe",
    "hi": "hin",
    "bn": "ben",
}

def load_tatoeba(seen_langs: List[str], unseen_langs: List[str], max_per_lang: int = 5000):
    """
    Load Tatoeba 'sentences' (text + language) and build splits:
      - train/validation/test from SEEN languages
      - ood_test from UNSEEN languages
    """
    ds = load_dataset("tatoeba", "sentences", split="train")
    seen_iso = {LANG_ALIAS.get(x, x) for x in seen_langs}
    unseen_iso = {LANG_ALIAS.get(x, x) for x in unseen_langs}

    def sample_lang(ds_iter, target_iso_set):
        bucket = {iso: [] for iso in target_iso_set}
        for row in ds_iter:
            lang = row.get("lang")
            if lang in bucket and len(bucket[lang]) < max_per_lang:
                bucket[lang].append({"text": row["text"], "label": lang})
            if all(len(v) >= max_per_lang for v in bucket.values()):
                break
        out = []
        for iso, items in bucket.items():
            out.extend(items)
        return out

    seen_data = sample_lang(ds, seen_iso)
    ds2 = load_dataset("tatoeba", "sentences", split="train")
    unseen_data = sample_lang(ds2, unseen_iso)

    random.seed(13)
    random.shuffle(seen_data)
    n = len(seen_data)
    n_train, n_val = int(0.8*n), int(0.9*n)
    train, val, test = seen_data[:n_train], seen_data[n_train:n_val], seen_data[n_val:]

    return {
        "train": train,
        "validation": val,
        "test": test,
        "ood_test": unseen_data
    }, sorted(list(seen_iso))

def load_flores(seen_langs: List[str], unseen_langs: List[str], max_per_lang: int = 2000):
    """
    Load FLORES-200 dev set and use text with ISO-639-3 codes.
    """
    ds = load_dataset("facebook/flores", "all", split="dev")
    seen_iso = {LANG_ALIAS.get(x, x) for x in seen_langs}
    unseen_iso = {LANG_ALIAS.get(x, x) for x in unseen_langs}

    def collect(ds_iter, target_iso, k):
        out = []
        counts = {iso: 0 for iso in target_iso}
        for row in ds_iter:
            lang = row["language"]
            if lang in counts and counts[lang] < k:
                out.append({"text": row["sentence"], "label": lang})
                counts[lang] += 1
            if all(v >= k for v in counts.values()):
                break
        return out

    seen_data = collect(ds, seen_iso, max_per_lang)
    ds2 = load_dataset("facebook/flores", "all", split="dev")
    unseen_data = collect(ds2, unseen_iso, max_per_lang)

    import random
    random.seed(13)
    random.shuffle(seen_data)
    n = len(seen_data)
    n_train, n_val = int(0.8*n), int(0.9*n)
    train, val, test = seen_data[:n_train], seen_data[n_train:n_val], seen_data[n_val:]

    return {
        "train": train,
        "validation": val,
        "test": test,
        "ood_test": unseen_data
    }, sorted(list(seen_iso))

def get_dataset(name: str, seen_langs: List[str], unseen_langs: List[str], max_per_lang: int = 5000):
    if name.lower() == "tatoeba":
        return load_tatoeba(seen_langs, unseen_langs, max_per_lang=max_per_lang)
    elif name.lower() == "flores":
        return load_flores(seen_langs, unseen_langs, max_per_lang=min(max_per_lang, 2000))
    elif name.lower() == "papluca":
        return load_papluca_langid(seen_langs, unseen_langs, max_per_lang=max_per_lang)
    else:
        raise ValueError(f"Unknown dataset: {name}")




def load_papluca_langid(seen_langs: List[str], unseen_langs: List[str], max_per_lang: int = 5000):
    """Load papluca/language-identification and build SEEN/UNSEEN splits.
    Expects columns: 'text' (str) and 'labels' (ISO-639-1 code like 'en').
    """
    from datasets import load_dataset
    ds = load_dataset("papluca/language-identification", split="train")

    seen_iso = set(seen_langs)
    unseen_iso = set(unseen_langs)

    def sample_lang(ds_iter, target, k):
        bucket = {iso: [] for iso in target}
        for row in ds_iter:
            lang = (row.get("labels") or "").strip()
            if lang in bucket and len(bucket[lang]) < k:
                text = (row.get("text") or "").strip()
                if text:
                    bucket[lang].append({"text": text, "label": lang})
            if all(len(v) >= k for v in bucket.values()):
                break
        out = []
        for iso, items in bucket.items():
            out.extend(items)
        return out

    seen_data = sample_lang(ds, seen_iso, max_per_lang)
    unseen_data = sample_lang(ds, unseen_iso, max_per_lang)

    import random
    random.seed(13)
    random.shuffle(seen_data)
    n = len(seen_data)
    n_train, n_val = int(0.8*n), int(0.9*n)
    train, val, test = seen_data[:n_train], seen_data[n_train:n_val], seen_data[n_val:]

    return {
        "train": train,
        "validation": val,
        "test": test,
        "ood_test": unseen_data
    }, sorted(list(seen_iso))
