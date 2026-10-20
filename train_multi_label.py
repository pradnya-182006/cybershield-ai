"""CyberShield AI - Multi-Label Model Training Pipeline (Phase 2).

Trains a six-label multi-label cyberbullying detection model using TF-IDF
features and one-vs-rest binary logistic classifiers.
Labels (in strict order):
    1. safe
    2. toxic
    3. insult
    4. threat
    5. obscene
    6. sarcasm

Designed for zero runtime dependencies (pure Python with optional scikit-learn
solver). Preserves existing legacy artifacts and saves a new versioned model
artifact to 'model/cybershield_v2.json'.
"""

import argparse
import datetime
import json
import math
import os
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from validate_dataset import DatasetValidator, ORDERED_LABELS

RANDOM_SEED = 42
DEFAULT_DATASET = "data/synthetic_comments_v2.json" if os.path.exists("data/synthetic_comments_v2.json") else "data/synthetic_comments.json"
DEFAULT_OUTPUT = "model/cybershield_v2.json"

# Calibrated classification thresholds per label.
# Sarcasm lowered to 0.35 to improve recall on this contextual, data-sparse label.
DEFAULT_THRESHOLDS = {
    "safe": 0.40,
    "toxic": 0.40,
    "insult": 0.40,
    "threat": 0.35,
    "obscene": 0.40,
    "sarcasm": 0.35,
}


# ── Sarcasm lexical cues ──────────────────────────────────────────────
_SARCASM_OPENER_WORDS = {
    "oh", "wow", "great", "fantastic", "wonderful", "brilliant", "amazing",
    "sure", "absolutely", "totally", "obviously", "clearly", "naturally",
    "right", "perfect", "charming", "delightful", "thrilling", "spectacular",
    "commendable", "inspiring", "kudos", "bravo", "congratulations",
    "superb", "splendid", "magnificent", "remarkable", "impressive",
    "refreshing", "lovely", "terrific", "truly",
}

_SARCASM_CONTEXT_WORDS = {
    "broken", "crashed", "delayed", "late", "wrong", "ruined", "missed",
    "failed", "useless", "pointless", "nothing", "nobody", "never",
    "worse", "worst", "outdated", "impossible", "wasted", "forgot",
    "empty", "zero", "no", "not", "complaint", "stuck",
    "broke", "deadline", "overtime", "delay", "restart", "error",
    "unfortunately", "again", "terrible",
}


def extract_features(text: str) -> List[str]:
    """Extract unigrams, bigrams, and salient structural/sarcasm cues from text."""
    text_lower = text.lower()
    words = re.findall(r"[a-z']+", text_lower)
    unigrams = words
    bigrams = [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]

    signals: List[str] = []
    if "!" in text:
        signals.append("__exclamation__")
    if "?" in text:
        signals.append("__question__")
    if len(text) > 5 and text.isupper():
        signals.append("__shouting__")

    # ── Sarcasm structural signals ─────────────────────────────────────
    word_set = set(words)

    # 1. Mock-praise opener: sentence starts with a sarcastic positive word
    if words and words[0] in _SARCASM_OPENER_WORDS:
        signals.append("__sarcasm_opener__")

    # 2. Contrast co-occurrence: mock-praise word AND negative-context word
    has_praise = bool(word_set & _SARCASM_OPENER_WORDS)
    has_negative = bool(word_set & _SARCASM_CONTEXT_WORDS)
    if has_praise and has_negative:
        signals.append("__sarcasm_contrast__")

    # 3. Rhetorical intensifier before negative word
    intensifiers = {"absolutely", "obviously", "clearly", "totally", "definitely", "naturally", "truly"}
    for i, w in enumerate(words[:-1]):
        if w in intensifiers and words[i + 1] in _SARCASM_CONTEXT_WORDS:
            signals.append("__sarcasm_intensifier__")
            break

    # 4. Comma-separated ironic constructions
    if "," in text and has_praise and has_negative:
        signals.append("__sarcasm_comma_irony__")

    # 5. 'because' used as rhetorical sarcastic connector
    if "because" in word_set and has_praise:
        signals.append("__sarcasm_because__")

    # 6. Passive-aggressive understatement markers
    understatement_phrases = [
        "as if", "yeah right", "oh sure", "of course", "right because",
        "love when", "love spending", "love receiving", "love being",
        "exactly what", "exactly how",
    ]
    for phrase in understatement_phrases:
        if phrase in text_lower:
            signals.append("__sarcasm_understatement__")
            break

    return unigrams + bigrams + signals


def sigmoid(z: float) -> float:
    """Stable sigmoid function bounded to avoid overflow."""
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def build_vocabulary(
    docs_features: List[List[str]], min_df: int = 1
) -> Tuple[Dict[str, int], List[float]]:
    """Build vocabulary and compute IDF values."""
    n_docs = len(docs_features)
    df = Counter()
    for f in docs_features:
        df.update(set(f))

    vocab_terms = sorted([term for term, count in df.items() if count >= min_df])
    vocab = {term: idx for idx, term in enumerate(vocab_terms)}
    idf = [math.log((1 + n_docs) / (1 + df[term])) + 1.0 for term in vocab_terms]
    return vocab, idf


def vectorize(
    features: List[str], vocab: Dict[str, int], idf: List[float]
) -> List[Tuple[int, float]]:
    """Transform token features into sub-linear TF-IDF normalized sparse vector."""
    tf = Counter(x for x in features if x in vocab)
    if not tf:
        return []
    vec = {vocab[t]: (1.0 + math.log(c)) * idf[vocab[t]] for t, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return [(i, v / norm) for i, v in vec.items()]


def train_logistic_regression(
    X: List[List[Tuple[int, float]]],
    y: List[int],
    vocab_size: int,
    epochs: int = 40,
    lr_init: float = 1.2,
    l2_reg: float = 0.0005,
    pos_weight: Optional[float] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[List[float], float]:
    """Train single binary logistic regression classifier with L2 regularization and class weighting."""
    rng = random.Random(seed)
    n = len(X)
    pos_count = sum(y)
    neg_count = n - pos_count

    if pos_weight is None:
        pos_weight = min(6.0, math.sqrt(neg_count / max(1, pos_count)) * 1.4)

    w = [0.0] * vocab_size
    b = -0.5 if pos_count < (n * 0.15) else 0.0

    indices = list(range(n))
    for ep in range(epochs):
        cur_lr = lr_init / (1.0 + 0.05 * ep)
        rng.shuffle(indices)
        for i in indices:
            vec = X[i]
            z = b + sum(w[j] * val for j, val in vec)
            pred = sigmoid(z)
            target = y[i]
            err = pred - target
            if target == 1:
                err *= pos_weight

            grad = cur_lr * err
            for j, val in vec:
                w[j] -= grad * val + cur_lr * l2_reg * w[j]
            b -= grad

    return w, b


def compute_metrics(
    y_true: List[List[int]],
    y_prob: List[List[float]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Compute precision, recall, F1, support, and subset accuracy across all labels."""
    n = len(y_true)
    per_label: Dict[str, Dict[str, Any]] = {}
    macro_f1s: List[float] = []
    tot_tp = tot_fp = tot_fn = tot_tn = 0
    exact_matches = 0
    total_label_errors = 0

    for k, label in enumerate(ORDERED_LABELS):
        thresh = thresholds.get(label, 0.40)
        tp = fp = fn = tn = 0
        for i in range(n):
            pred = 1 if y_prob[i][k] >= thresh else 0
            true = y_true[i][k]
            if pred == 1 and true == 1:
                tp += 1
            elif pred == 1 and true == 0:
                fp += 1
                total_label_errors += 1
            elif pred == 0 and true == 1:
                fn += 1
                total_label_errors += 1
            else:
                tn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp + fn

        macro_f1s.append(f1)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_tn += tn

        per_label[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "threshold": thresh,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    for i in range(n):
        pred_vec = [1 if y_prob[i][k] >= thresholds.get(ORDERED_LABELS[k], 0.40) else 0 for k in range(6)]
        if pred_vec == y_true[i]:
            exact_matches += 1

    micro_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
    subset_acc = exact_matches / n if n > 0 else 0.0
    hamming_loss = total_label_errors / (n * len(ORDERED_LABELS)) if n > 0 else 0.0

    return {
        "per_label": per_label,
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "subset_accuracy": round(subset_acc, 4),
        "exact_matches": exact_matches,
        "hamming_loss": round(hamming_loss, 4),
        "total_samples": n,
    }


def evaluate_cross_validation(
    dataset: List[Dict[str, Any]],
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Perform 5-fold cross-validation with zero data leakage."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    rng = random.Random(seed)
    n = len(dataset)
    indices = list(range(n))
    rng.shuffle(indices)

    folds = [indices[i::n_splits] for i in range(n_splits)]
    cv_prob = [[0.0] * len(ORDERED_LABELS) for _ in range(n)]

    y_true = []
    for d in dataset:
        l_set = set(str(l).strip().lower() for l in d.get("labels", []))
        y_true.append([1 if l in l_set else 0 for l in ORDERED_LABELS])

    for fold_num, val_indices in enumerate(folds, start=1):
        val_set = set(val_indices)
        train_indices = [idx for idx in indices if idx not in val_set]

        # Extract features and build vocab strictly from training split
        train_feats = [extract_features(dataset[idx]["comment"]) for idx in train_indices]
        fold_vocab, fold_idf = build_vocabulary(train_feats, min_df=1)

        X_train = [vectorize(f, fold_vocab, fold_idf) for f in train_feats]

        # Train one classifier per label
        for k, label in enumerate(ORDERED_LABELS):
            y_train_k = [y_true[idx][k] for idx in train_indices]
            w_k, b_k = train_logistic_regression(
                X_train,
                y_train_k,
                vocab_size=len(fold_vocab),
                epochs=40,
                lr_init=1.2,
                seed=seed + fold_num + k,
            )

            # Evaluate on held-out validation samples
            for val_idx in val_indices:
                val_feats = extract_features(dataset[val_idx]["comment"])
                val_vec = vectorize(val_feats, fold_vocab, fold_idf)
                z = b_k + sum(w_k[j] * val for j, val in val_vec)
                cv_prob[val_idx][k] = sigmoid(z)

    return compute_metrics(y_true, cv_prob, thresholds)


def evaluate_train_test_split(
    dataset: List[Dict[str, Any]],
    test_ratio: float = 0.20,
    seed: int = RANDOM_SEED,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Perform a held-out 80/20 train/test split evaluation."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    rng = random.Random(seed)
    n = len(dataset)
    indices = list(range(n))
    rng.shuffle(indices)

    split_point = int(n * (1.0 - test_ratio))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    y_all = []
    for d in dataset:
        l_set = set(str(l).strip().lower() for l in d.get("labels", []))
        y_all.append([1 if l in l_set else 0 for l in ORDERED_LABELS])

    train_feats = [extract_features(dataset[idx]["comment"]) for idx in train_indices]
    test_feats = [extract_features(dataset[idx]["comment"]) for idx in test_indices]

    split_vocab, split_idf = build_vocabulary(train_feats, min_df=1)
    X_train = [vectorize(f, split_vocab, split_idf) for f in train_feats]
    X_test = [vectorize(f, split_vocab, split_idf) for f in test_feats]

    test_prob = [[0.0] * len(ORDERED_LABELS) for _ in range(len(test_indices))]
    y_test = [y_all[idx] for idx in test_indices]

    for k, label in enumerate(ORDERED_LABELS):
        y_train_k = [y_all[idx][k] for idx in train_indices]
        w_k, b_k = train_logistic_regression(
            X_train,
            y_train_k,
            vocab_size=len(split_vocab),
            epochs=40,
            lr_init=1.2,
            seed=seed + k,
        )

        for i, test_vec in enumerate(X_test):
            z = b_k + sum(w_k[j] * val for j, val in test_vec)
            test_prob[i][k] = sigmoid(z)

    return compute_metrics(y_test, test_prob, thresholds)


def train_full_model(
    dataset: List[Dict[str, Any]],
    seed: int = RANDOM_SEED,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Train the final production model on all 100 comments and return full serializable artifact."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    docs_features = [extract_features(d["comment"]) for d in dataset]
    vocab, idf = build_vocabulary(docs_features, min_df=1)
    X = [vectorize(f, vocab, idf) for f in docs_features]

    y_all = []
    for d in dataset:
        l_set = set(str(l).strip().lower() for l in d.get("labels", []))
        y_all.append([1 if l in l_set else 0 for l in ORDERED_LABELS])

    W: Dict[str, List[float]] = {}
    B: Dict[str, float] = {}

    for k, label in enumerate(ORDERED_LABELS):
        y_k = [y_all[i][k] for i in range(len(dataset))]
        w_k, b_k = train_logistic_regression(
            X,
            y_k,
            vocab_size=len(vocab),
            epochs=45,
            lr_init=1.2,
            seed=seed + k,
        )
        W[label] = [round(x, 5) for x in w_k]
        B[label] = round(b_k, 5)

    # Compute training set fit
    full_probs = []
    for vec in X:
        row_prob = [sigmoid(B[lab] + sum(W[lab][j] * val for j, val in vec)) for lab in ORDERED_LABELS]
        full_probs.append(row_prob)

    train_fit_metrics = compute_metrics(y_all, full_probs, thresholds)

    artifact = {
        "version": "2.0.0",
        "labels": ORDERED_LABELS,
        "thresholds": thresholds,
        "vocab": vocab,
        "idf": [round(x, 5) for x in idf],
        "w": W,
        "b": B,
        "metadata": {
            "model_type": "Multi-Label One-vs-Rest Logistic Regression (TF-IDF)",
            "dataset_file": "data/synthetic_comments.json",
            "num_samples": len(dataset),
            "vocabulary_size": len(vocab),
            "random_seed": seed,
            "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_fit": {
                "subset_accuracy": train_fit_metrics["subset_accuracy"],
                "macro_f1": train_fit_metrics["macro_f1"],
                "micro_f1": train_fit_metrics["micro_f1"],
                "hamming_loss": train_fit_metrics["hamming_loss"],
            },
            "limitations_notice": (
                "Prototype model trained on 100 synthetic comments. "
                "Useful for baseline validation and architecture verification. "
                "Will require larger real-world datasets for high out-of-domain generalization."
            ),
        },
    }

    return artifact, train_fit_metrics


def print_evaluation_report(
    cv_metrics: Dict[str, Any],
    split_metrics: Dict[str, Any],
    train_fit: Dict[str, Any],
) -> None:
    """Print readable evaluation tables to console."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print(" CyberShield AI - Phase 2 Multi-Label Model Evaluation")
    print("=" * 70)
    print("\n[Method 1] 5-Fold Cross Validation (Held-out across all 100 samples, 0 Leakage):")
    print("-" * 70)
    print(f"{'Label':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Threshold':<11} {'Support':<8}")
    print("-" * 70)
    for lbl in ORDERED_LABELS:
        m = cv_metrics["per_label"][lbl]
        print(f"{lbl:<10} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1_score']:<10.3f} {m['threshold']:<11.2f} {m['support']:<8}")
    print("-" * 70)
    print(f"Macro F1-Score : {cv_metrics['macro_f1']:.3f} | Micro F1: {cv_metrics['micro_f1']:.3f}")
    print(f"Subset Accuracy: {cv_metrics['exact_matches']}/{cv_metrics['total_samples']} ({cv_metrics['subset_accuracy']*100:.1f}%) | Hamming Loss: {cv_metrics['hamming_loss']:.3f}")

    print("\n[Method 2] 80/20 Train/Test Split (20 Held-Out Test Samples):")
    print("-" * 70)
    print(f"{'Label':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Threshold':<11} {'Support':<8}")
    print("-" * 70)
    for lbl in ORDERED_LABELS:
        m = split_metrics["per_label"][lbl]
        print(f"{lbl:<10} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1_score']:<10.3f} {m['threshold']:<11.2f} {m['support']:<8}")
    print("-" * 70)
    print(f"Macro F1-Score : {split_metrics['macro_f1']:.3f} | Micro F1: {split_metrics['micro_f1']:.3f}")
    print(f"Subset Accuracy: {split_metrics['exact_matches']}/{split_metrics['total_samples']} ({split_metrics['subset_accuracy']*100:.1f}%) | Hamming Loss: {split_metrics['hamming_loss']:.3f}")

    print("\n[Method 3] Full Training Set Convergence (Reference Fit):")
    print(f"Macro F1: {train_fit['macro_f1']:.3f} | Micro F1: {train_fit['micro_f1']:.3f} | Subset Accuracy: {train_fit['subset_accuracy']*100:.1f}%")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train CyberShield AI 6-Label Multi-Label Model.")
    parser.add_argument(
        "--dataset",
        "-d",
        default=DEFAULT_DATASET,
        help=f"Path to validated synthetic dataset. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"Path to save new versioned model artifact. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility. Default: {RANDOM_SEED}",
    )
    args = parser.parse_args()

    # Step 1: Validate dataset
    validator = DatasetValidator()
    records, load_errors = validator.load_dataset(args.dataset)
    if load_errors:
        print(f"[FAIL] Could not load dataset from '{args.dataset}': {load_errors}")
        sys.exit(1)

    val_stats = validator.validate_dataset(records)
    if val_stats["invalid_records"] > 0:
        print(f"[FAIL] Dataset contains {val_stats['invalid_records']} invalid records. Fix validation errors first.")
        sys.exit(1)

    print(f"[INFO] Loaded and validated {len(records)} records from {args.dataset}.")

    # Step 2: Honest Evaluation
    print("[INFO] Evaluating model with 5-fold cross validation...")
    cv_metrics = evaluate_cross_validation(records, n_splits=5, seed=args.seed)

    print("[INFO] Evaluating model with 80/20 train/test split...")
    split_metrics = evaluate_train_test_split(records, test_ratio=0.20, seed=args.seed)

    # Step 3: Train full model artifact
    print(f"[INFO] Training final model artifact across all {len(records)} comments...")
    artifact, train_fit = train_full_model(records, seed=args.seed)

    # Attach evaluation results to artifact metadata
    artifact["metadata"]["cv_evaluation"] = {
        "macro_f1": cv_metrics["macro_f1"],
        "micro_f1": cv_metrics["micro_f1"],
        "subset_accuracy": cv_metrics["subset_accuracy"],
        "per_label": {k: {"f1": v["f1_score"], "precision": v["precision"], "recall": v["recall"], "support": v["support"]} for k, v in cv_metrics["per_label"].items()},
    }
    artifact["metadata"]["split_evaluation"] = {
        "macro_f1": split_metrics["macro_f1"],
        "micro_f1": split_metrics["micro_f1"],
        "subset_accuracy": split_metrics["subset_accuracy"],
        "per_label": {k: {"f1": v["f1_score"], "precision": v["precision"], "recall": v["recall"], "support": v["support"]} for k, v in split_metrics["per_label"].items()},
    }

    # Save to versioned artifact path (preserving legacy cybershield.json!)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"[SUCCESS] Model artifact successfully saved to: {args.output}")

    # Print summary
    print_evaluation_report(cv_metrics, split_metrics, train_fit)


if __name__ == "__main__":
    main()
