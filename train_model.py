"""CyberShield AI - model training (pure Python: TF-IDF + logistic regression).
No numpy / pandas / scikit-learn needed, so Windows Application Control cannot block it."""
import csv
import json
import math
import os
import random
import re
from collections import Counter

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
DATA_PATH = "data/train.csv"
MODEL_PATH = "model/cybershield.json"
MAX_ROWS = 60000      # rows used from train.csv (raise for better accuracy, but slower)
MAX_FEATURES = 30000
EPOCHS = 4

SAMPLE = [
    ("you are so stupid and worthless", ["toxic", "insult"]),
    ("i will find you and hurt you", ["toxic", "threat"]),
    ("shut up you idiot nobody likes you", ["toxic", "insult"]),
    ("go kill yourself loser", ["toxic", "severe_toxic", "threat", "insult"]),
    ("what the hell is this garbage", ["toxic", "obscene"]),
    ("you are a disgusting pig", ["toxic", "insult"]),
    ("you people are disgusting and should go back where you came from", ["toxic", "identity_hate"]),
    ("i hate your kind you do not belong here", ["toxic", "identity_hate"]),
    ("you are a pathetic ugly freak", ["toxic", "insult"]),
    ("i will destroy you after school", ["toxic", "threat"]),
    ("this is damn bullshit", ["toxic", "obscene"]),
    ("you are dumb and nobody wants you here", ["toxic", "insult"]),
    ("great job on the project, well done", []),
    ("thanks for helping me with my homework", []),
    ("i disagree with your point but i respect your view", []),
    ("have a wonderful day my friend", []),
    ("congratulations on your new job", []),
    ("can you explain how this works please", []),
    ("i really enjoyed the movie last night", []),
    ("let us meet tomorrow to finish the assignment", []),
    ("that was a thoughtful and kind comment", []),
    ("good luck with your exam you will do great", []),
    ("the weather is lovely today", []),
    ("nice work team keep it up", []),
]


def features(text):
    words = re.findall(r"[a-z']+", text.lower())
    return words + [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]


def load_data():
    rows = []
    if os.path.exists(DATA_PATH):
        print("Loading dataset:", DATA_PATH)
        with open(DATA_PATH, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((r["comment_text"], [int(r[l]) for l in LABELS]))
                except (KeyError, ValueError, TypeError):
                    continue
        random.shuffle(rows)
        return rows[:MAX_ROWS]
    print("train.csv not found. Using small built-in sample (low accuracy).")
    for _ in range(6):
        for text, labs in SAMPLE:
            rows.append((text, [1 if l in labs else 0 for l in LABELS]))
    random.shuffle(rows)
    return rows


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def main():
    random.seed(42)
    data = load_data()
    n = len(data)
    print("Training rows:", n)
    min_df = 3 if n > 1000 else 1

    # ---- TF-IDF vocabulary ----
    df = Counter()
    docs = []
    for text, _ in data:
        f = features(text)
        docs.append(f)
        df.update(set(f))
    terms = [t for t, c in df.most_common(MAX_FEATURES) if c >= min_df]
    vocab = {t: i for i, t in enumerate(terms)}
    idf = [math.log((1 + n) / (1 + df[t])) + 1.0 for t in terms]

    def vectorize(feats):
        tf = Counter(x for x in feats if x in vocab)
        vec = {vocab[t]: (1 + math.log(c)) * idf[vocab[t]] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return [(i, v / norm) for i, v in vec.items()]

    X = [vectorize(f) for f in docs]
    Y = [y for _, y in data]
    split = int(n * 0.9)
    tr, te = list(range(split)), list(range(split, n))

    # ---- logistic regression (SGD), one model per label ----
    V = len(terms)
    W = {l: [0.0] * V for l in LABELS}
    B = {l: 0.0 for l in LABELS}
    for k, lab in enumerate(LABELS):
        pos = sum(Y[i][k] for i in tr)
        posw = min(8.0, math.sqrt(max(1, len(tr) - pos) / max(1, pos)))
        w, b = W[lab], 0.0
        for ep in range(EPOCHS):
            random.shuffle(tr)
            lr = 0.9 / (1 + ep)
            for i in tr:
                vec = X[i]
                z = b + sum(w[j] * v for j, v in vec)
                err = sigmoid(z) - Y[i][k]
                if Y[i][k]:
                    err *= posw
                g = lr * err
                for j, v in vec:
                    w[j] -= g * v
                b -= g
        B[lab] = b
        print("trained:", lab)

    # ---- evaluation ----
    tp = fp = fn = correct = total = 0
    for i in te:
        for k, lab in enumerate(LABELS):
            p = 1 if sigmoid(B[lab] + sum(W[lab][j] * v for j, v in X[i])) >= 0.5 else 0
            y = Y[i][k]
            tp += p and y
            fp += p and not y
            fn += (not p) and y
            correct += p == y
            total += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print("Accuracy :", round(correct / total, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1       :", round(f1, 4))

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "vocab": vocab, "idf": idf,
                   "w": {l: [round(x, 5) for x in W[l]] for l in LABELS}, "b": B}, f)
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    main()