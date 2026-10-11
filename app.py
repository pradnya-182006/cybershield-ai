import json
import math
import os
import re
import sys
import threading
import uuid
from collections import Counter
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = "model/cybershield.json"
HISTORY_PATH = "data/history.json"
lock = threading.Lock()

if not os.path.exists(MODEL_PATH):
    sys.exit("Model not found. Run:  python train_model.py  first.")

with open(MODEL_PATH, "r", encoding="utf-8") as f:
    M = json.load(f)
LABELS, VOCAB, IDF, W, B = M["labels"], M["vocab"], M["idf"], M["w"], M["b"]

STOP_WORDS = set("a an the and or but if of to in on at for with is are was were be been it its this that "
                 "i you he she we they me my your his her our their am do does did so as by from not no "
                 "will would can could just than then there here what who how when where why".split())


def features(text):
    words = re.findall(r"[a-z']+", text.lower())
    return words + [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def vectorize(text):
    tf = Counter(x for x in features(text) if x in VOCAB)
    vec = {VOCAB[t]: (1 + math.log(c)) * IDF[VOCAB[t]] for t, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {i: v / norm for i, v in vec.items()}


def predict(vec):
    return {lab: sigmoid(B[lab] + sum(W[lab][i] * v for i, v in vec.items())) for lab in LABELS}


REASONS = {
    "toxic": "Rude or disrespectful language",
    "severe_toxic": "Extremely hostile language",
    "obscene": "Obscene or vulgar language",
    "threat": "Language that may threaten harm",
    "insult": "Insulting or demeaning term",
    "identity_hate": "Language targeting a group or identity",
}

# Small lexicon that backs up the ML model for common abusive words
LEXICON = {
    "stupid": "insult", "idiot": "insult", "dumb": "insult", "loser": "insult",
    "worthless": "insult", "pathetic": "insult", "ugly": "insult", "freak": "insult",
    "moron": "insult", "trash": "insult", "kill": "threat", "hurt": "threat",
    "destroy": "threat", "die": "threat", "shit": "obscene", "damn": "obscene",
    "hell": "obscene", "bullshit": "obscene", "disgusting": "toxic", "hate": "toxic",
}

REPLACEMENTS = {
    "stupid": "mistaken", "idiot": "thoughtless person", "dumb": "unwise", "loser": "person",
    "worthless": "struggling", "pathetic": "unfortunate", "ugly": "different",
    "moron": "thoughtless person", "trash": "poor", "disgusting": "unpleasant", "hate": "dislike",
}


def explain(text):
    """Find the words that pushed the model toward a harmful label."""
    vec = vectorize(text)
    inv = {i: t for t, i in VOCAB.items() if i in vec}
    found = {}
    for idx, val in vec.items():
        term = inv[idx]
        if " " in term or (term in STOP_WORDS and term not in LEXICON):
            continue
        best, best_label = 0.0, None
        for lab in LABELS:
            c = W[lab][idx] * val
            if c > best:
                best, best_label = c, lab
        if best_label and best > 0.4:
            found[term] = (min(1.0, best / 1.5), best_label)
    for tok in re.findall(r"\b\w+\b", text.lower()):
        if tok in LEXICON and tok not in found:
            found[tok] = (0.8, LEXICON[tok])
    return [
        {"word": w, "weight": round(v[0], 2), "reason": REASONS[v[1]]}
        for w, v in sorted(found.items(), key=lambda x: -x[1][0])
    ][:10]


def get_severity(scores):
    top = max(scores.values())
    if scores["threat"] > 0.6 or scores["severe_toxic"] > 0.6 or top >= 0.8:
        return "high"
    if top >= 0.65:
        return "medium"
    if top >= 0.5:
        return "low"
    return "none"


def safer_rewrite(text, flagged, scores):
    if max(scores.values()) < 0.35:
        return text
    if scores["threat"] > 0.5 or scores["severe_toxic"] > 0.5:
        return "I'm very frustrated right now. Can we talk about this calmly and find a solution together?"
    if len([f for f in flagged if f["word"] in REPLACEMENTS or f["word"] in LEXICON]) >= 2:
        return "I disagree with this and would like to discuss it respectfully."
    out = text
    for f in flagged:
        pattern = re.compile(r"\b" + re.escape(f["word"]) + r"\b", re.IGNORECASE)
        out = pattern.sub(REPLACEMENTS.get(f["word"], ""), out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\b(a|an) (\w)", lambda m: ("an " if m.group(2).lower() in "aeiou" else "a ") + m.group(2), out, flags=re.I)
    if len(out.split()) < 4:
        return "I disagree with this and would like to discuss it respectfully."
    return out


def tone_tags(text, scores):
    tags = []
    if len(text) > 5 and text.isupper():
        tags.append("Shouting")
    if scores["threat"] > 0.5:
        tags.append("Threatening")
    if scores["insult"] > 0.5:
        tags.append("Demeaning")
    if scores["obscene"] > 0.5:
        tags.append("Vulgar")
    if scores["identity_hate"] > 0.5:
        tags.append("Discriminatory")
    if text.count("!") >= 3:
        tags.append("Aggressive")
    return tags


def analyze_text(text):
    text = text.strip()[:1000]
    vec = vectorize(text)
    probs = predict(vec)
    scores = {lab: round(p, 3) for lab, p in probs.items()}
    flagged = explain(text)

    # no known words, or no harmful word found and model not very sure -> treat as safe
    if not vec or (not flagged and max(scores.values()) < 0.8):
        scores = {lab: min(p, 0.2) for lab, p in scores.items()}

    # lexicon boost so obvious abuse is never missed
    for f in flagged:
        lab = LEXICON.get(f["word"])
        if lab and scores.get(lab, 0) < 0.6:
            scores[lab] = 0.6
            scores["toxic"] = max(scores["toxic"], 0.6)

    top_score = max(scores.values())
    top_label = max(scores, key=scores.get)
    severity = get_severity(scores)
    harmful = severity != "none"
    label = top_label if harmful else "safe"

    if harmful:
        reasons = sorted({f["reason"] for f in flagged})
        explanation = (
            f"This message was flagged as '{label.replace('_', ' ')}' with {severity} severity. "
            + (("Reasons: " + "; ".join(reasons) + ". ") if reasons else "")
            + "The highlighted words contributed most to this result."
        )
    else:
        explanation = "No harmful or bullying language was detected in this message."

    return {
        "label": label,
        "scores": scores,
        "overall_score": round(top_score, 3),
        "severity": severity,
        "confidence": round(abs(top_score - 0.5) * 2, 2),
        "flagged_words": flagged if harmful else [],
        "explanation": explanation,
        "safer_rewrite": safer_rewrite(text, flagged, scores),
        "tone_tags": tone_tags(text, scores),
        "disclaimer": "Predictions are estimates. Context, sarcasm and culture can change meaning.",
    }


# ---------- history helpers ----------
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_history(items):
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


# ---------- pages ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ---------- API ----------
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."}), 400
    result = analyze_text(text)
    with lock:
        items = load_history()
        items.append({
            "id": str(uuid.uuid4())[:8],
            "text": text[:200],
            "label": result["label"],
            "severity": result["severity"],
            "score": result["overall_score"],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        save_history(items[-500:])
    return jsonify(result)


@app.route("/api/history", methods=["GET", "DELETE"])
def api_history():
    with lock:
        if request.method == "DELETE":
            save_history([])
            return jsonify({"ok": True})
        return jsonify(list(reversed(load_history()))[:50])


@app.route("/api/stats")
def api_stats():
    h = load_history()
    total = len(h)
    harmful = [r for r in h if r["label"] != "safe"]
    high = [r for r in h if r["severity"] == "high"]
    today = datetime.now().date()
    days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    per_day = Counter(r["time"][:10] for r in h)
    toxic_per_day = Counter(r["time"][:10] for r in harmful)
    return jsonify({
        "total": total,
        "toxic_pct": round(len(harmful) / total * 100, 1) if total else 0,
        "safe_pct": round((total - len(harmful)) / total * 100, 1) if total else 0,
        "high_count": len(high),
        "categories": dict(Counter(r["label"] for r in harmful)),
        "severity": {s: sum(1 for r in h if r["severity"] == s) for s in ["none", "low", "medium", "high"]},
        "trend": {
            "labels": [d.strftime("%d %b") for d in days],
            "total": [per_day.get(d.isoformat(), 0) for d in days],
            "harmful": [toxic_per_day.get(d.isoformat(), 0) for d in days],
        },
    })


if __name__ == "__main__":
    app.run(debug=True)