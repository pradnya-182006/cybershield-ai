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

# Agent pipeline (Phase 3)
try:
    from agent import run_agent_pipeline
    AGENT_AVAILABLE = True
except Exception as _agent_err:
    print(f"[WARN] Agent pipeline unavailable: {_agent_err}")
    AGENT_AVAILABLE = False
    run_agent_pipeline = None

app = Flask(__name__)

# Model configuration: CYBERSHIELD_MODEL_VERSION can be "v2" (default) or "v1" (legacy rollback)
MODEL_V2_PATH = "model/cybershield_v2.json"
MODEL_LEGACY_PATH = "model/cybershield.json"
HISTORY_PATH = "data/history.json"
lock = threading.Lock()

ACTIVE_MODEL_VERSION = os.environ.get("CYBERSHIELD_MODEL_VERSION", "v2").lower()
USE_V2 = (ACTIVE_MODEL_VERSION == "v2") and os.path.exists(MODEL_V2_PATH)

predictor_v2 = None
if USE_V2:
    try:
        from predictor import get_predictor
        predictor_v2 = get_predictor(MODEL_V2_PATH)
        print(f"[INFO] CyberShield AI running with Model V2: {MODEL_V2_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load Model V2 ({e}). Falling back to legacy model.")
        USE_V2 = False

# Fallback / Legacy model loading (guaranteed to preserve legacy behavior)
if not USE_V2:
    print(f"[INFO] CyberShield AI running in Legacy V1 mode: {MODEL_LEGACY_PATH}")
    if not os.path.exists(MODEL_LEGACY_PATH):
        sys.exit(f"Legacy model not found at {MODEL_LEGACY_PATH}.")
    with open(MODEL_LEGACY_PATH, "r", encoding="utf-8") as f:
        M = json.load(f)
    LABELS, VOCAB, IDF, W, B = M["labels"], M["vocab"], M["idf"], M["w"], M["b"]
else:
    # Set default labels for V2
    LABELS = predictor_v2.labels
    if os.path.exists(MODEL_LEGACY_PATH):
        try:
            with open(MODEL_LEGACY_PATH, "r", encoding="utf-8") as f:
                M_leg = json.load(f)
            VOCAB, IDF, W, B = M_leg["vocab"], M_leg["idf"], M_leg["w"], M_leg["b"]
        except Exception:
            VOCAB, IDF, W, B = {}, [], {}, {}
    else:
        VOCAB, IDF, W, B = {}, [], {}, {}

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
    # Use W.keys() to reference only labels that exist in the loaded weight dict.
    # When V2 is active, LABELS = V2 labels but W/B come from the legacy model
    # (which only has Kaggle labels). Using W.keys() prevents KeyError on 'safe' etc.
    return {lab: sigmoid(B[lab] + sum(W[lab][i] * v for i, v in vec.items())) for lab in W}


REASONS = {
    "safe": "Constructive, harmless language",
    "toxic": "Rude, hostile or disrespectful language",
    "insult": "Insulting or demeaning term",
    "threat": "Language that threatens physical harm or violence",
    "obscene": "Obscene or vulgar language",
    "sarcasm": "Mocking, ironic or passive-aggressive tone",
    # Legacy Kaggle labels for backward compatibility:
    "severe_toxic": "Extremely hostile language",
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
        # Use W.keys() — legacy W only has Kaggle labels, not V2 labels like 'safe'
        for lab in W:
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
    threat_p = scores.get("threat", 0.0)
    severe_p = scores.get("severe_toxic", 0.0)
    top = max(scores.values()) if scores else 0.0
    if threat_p > 0.35 or severe_p > 0.50 or top >= 0.80:
        return "high"
    if top >= 0.65:
        return "medium"
    if top >= 0.40:
        return "low"
    return "none"


def safer_rewrite(text, flagged, scores):
    if max(scores.values()) < 0.35:
        return text
    if scores.get("threat", 0.0) > 0.35 or scores.get("severe_toxic", 0.0) > 0.5:
        return "I'm very frustrated right now. Can we talk about this calmly and find a solution together?"
    if scores.get("sarcasm", 0.0) > 0.45 and scores.get("toxic", 0.0) < 0.40:
        return "I have concerns regarding this matter and would like to address them directly and constructively."
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
    if scores.get("threat", 0.0) > 0.35:
        tags.append("Threatening")
    if scores.get("insult", 0.0) > 0.40:
        tags.append("Demeaning")
    if scores.get("obscene", 0.0) > 0.40:
        tags.append("Vulgar")
    if scores.get("sarcasm", 0.0) > 0.35:
        tags.append("Sarcastic")
    if scores.get("identity_hate", 0.0) > 0.50:
        tags.append("Discriminatory")
    if text.count("!") >= 3:
        tags.append("Aggressive")
    return tags


def analyze_text(text):
    text = text.strip()[:1000]

    # Model V2 Path
    if USE_V2 and predictor_v2 is not None:
        pred_res = predictor_v2.predict(text)
        scores = pred_res["scores"]
        top_label = pred_res["primary_label"]
        severity = pred_res["severity"]
        confidence = pred_res["confidence"]
        harmful = severity != "none"
        is_uncertain = bool(
            pred_res.get("uncertain", False)
            or top_label == "uncertain"
            or (not harmful and scores.get("safe", 0.0) < 0.35)
        )

        if harmful:
            label = top_label
        elif is_uncertain:
            label = "uncertain"
        else:
            label = "safe"

        harmful_keys = [l for l in ["toxic", "insult", "threat", "obscene"] if l in scores]
        harmfulness_score = max([scores[l] for l in harmful_keys]) if (harmful and harmful_keys) else 0.0

        flagged = explain(text) if (not is_uncertain and not pred_res.get("uncertain", False)) else []
        if harmful:
            reasons = sorted({f["reason"] for f in flagged if "reason" in f})
            policy_notice = " [Safety Notice: Flagged for moderator review; never auto-ban.]" if label == "threat" else ""
            explanation = (
                f"This message was flagged as '{label.replace('_', ' ')}' with {severity} severity.{policy_notice} "
                + (("Reasons: " + "; ".join(reasons) + ". ") if reasons else "")
                + "The highlighted words contributed most to this result."
            )
        elif is_uncertain:
            explanation = "The model could not confidently classify this comment. No automatic moderation action was taken."
        else:
            if pred_res.get("primary_label") == "sarcasm":
                explanation = "Contextual sarcasm detected without active toxicity. Sarcasm is context-dependent and reviewed separately."
            else:
                explanation = "No harmful or bullying language was detected in this message."

        # Actual probability of the predicted label
        if label in scores:
            label_confidence = round(scores[label], 4)
        elif label == "uncertain":
            label_confidence = round(scores.get("safe", 0.0), 4)
        elif label == "safe":
            label_confidence = round(scores.get("safe", 0.0), 4)
        else:
            label_confidence = round(scores.get(label, 0.0), 4)

        return {
            "label": label,
            "scores": scores,
            "label_confidence": label_confidence,
            "overall_score": round(confidence, 3),
            "harmfulness_score": round(harmfulness_score, 3),
            "severity": severity,
            "confidence": round(abs(confidence - 0.5) * 2, 2),
            "uncertain": is_uncertain,
            "flagged_words": flagged if harmful else [],
            "explanation": explanation,
            "safer_rewrite": safer_rewrite(text, flagged, scores),
            "tone_tags": tone_tags(text, scores),
            "predicted_labels": pred_res["predicted_labels"],
            "moderation_policy": pred_res["moderation_policy"],
            "auxiliary_context": pred_res.get("auxiliary_context"),
            "model_version": pred_res["version"],
            "disclaimer": "Predictions are estimates. Context, sarcasm and culture can change meaning.",
        }

    # Legacy V1 Path (Rollback / Fallback)
    vec = vectorize(text)
    probs = predict(vec)
    scores = {lab: round(p, 3) for lab, p in probs.items()}
    flagged = explain(text)

    if not vec or (not flagged and max(scores.values()) < 0.8):
        scores = {lab: min(p, 0.2) for lab, p in scores.items()}

    for f in flagged:
        lab = LEXICON.get(f["word"])
        if lab and scores.get(lab, 0) < 0.6:
            scores[lab] = 0.6
            scores["toxic"] = max(scores["toxic"], 0.6)

    # In V1 rollback path, scores dict is keyed by legacy labels (from W.keys())
    top_score = max(scores.values()) if scores else 0.0
    top_label = max(scores, key=scores.get) if scores else "safe"
    severity = get_severity(scores)
    harmful = severity != "none"
    is_uncertain = (not vec) or (not harmful and top_score < 0.30)

    if harmful:
        label = top_label
    elif is_uncertain:
        label = "uncertain"
    else:
        label = "safe"

    if harmful:
        reasons = sorted({f["reason"] for f in flagged})
        explanation = (
            f"This message was flagged as '{label.replace('_', ' ')}' with {severity} severity. "
            + (("Reasons: " + "; ".join(reasons) + ". ") if reasons else "")
            + "The highlighted words contributed most to this result."
        )
    elif is_uncertain:
        explanation = "The model could not confidently classify this comment. No automatic moderation action was taken."
    else:
        explanation = "No harmful or bullying language was detected in this message."

    if label in scores:
        label_confidence = round(scores[label], 4)
    elif label in ("safe", "uncertain"):
        label_confidence = round(max(0.0, 1.0 - top_score), 4)
    else:
        label_confidence = round(scores.get(label, 0.0), 4)

    return {
        "label": label,
        "scores": scores,
        "label_confidence": label_confidence,
        "overall_score": round(top_score, 3),
        "harmfulness_score": round(top_score if harmful else 0.0, 3),
        "severity": severity,
        "confidence": round(abs(top_score - 0.5) * 2, 2),
        "uncertain": is_uncertain,
        "flagged_words": flagged if harmful else [],
        "explanation": explanation,
        "safer_rewrite": safer_rewrite(text, flagged, scores),
        "tone_tags": tone_tags(text, scores),
        "predicted_labels": [label] if harmful else ([] if is_uncertain else ["safe"]),
        "model_version": "1.0.0-legacy",
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

    try:
        result = analyze_text(text)
    except Exception:
        # Prevent stack trace leakage to clients
        return jsonify({"error": "An error occurred during analysis. Please try again."}), 500

    with lock:
        items = load_history()
        items.append({
            "id": str(uuid.uuid4())[:8],
            "text": text[:200],
            "label": result["label"],
            "severity": result["severity"],
            "score": result["harmfulness_score"],
            "uncertain": result.get("uncertain", False),
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
    harmful = [r for r in h if r.get("label") not in ("safe", "uncertain") and r.get("severity") != "none"]
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


@app.route("/api/agent", methods=["POST"])
def api_agent():
    """Run the full 4-stage agentic moderation pipeline on a comment."""
    if not AGENT_AVAILABLE or run_agent_pipeline is None:
        return jsonify({"error": "Agent pipeline is not available."}), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    try:
        # Run full analyze first to get predictor result
        full_result = analyze_text(text)

        # Build predictor_result dict compatible with agent pipeline
        # (predictor_result must have 'scores', 'predicted_labels', 'severity', etc.)
        if USE_V2 and predictor_v2 is not None:
            pred_res = predictor_v2.predict(text)
        else:
            # Build a synthetic predictor_result from legacy analysis
            scores = full_result.get("scores", {})
            label = full_result.get("label", "safe")
            severity = full_result.get("severity", "none")
            pred_res = {
                "scores": scores,
                "predicted_labels": full_result.get("predicted_labels", [label]),
                "raw_predicted_labels": full_result.get("predicted_labels", [label]),
                "primary_label": label,
                "severity": severity,
                "confidence": full_result.get("confidence", 0.0),
                "uncertain": False,
                "moderation_policy": full_result.get("moderation_policy", {
                    "action": "ALLOW",
                    "priority": "NONE",
                    "reason": "Legacy model result.",
                }),
                "auxiliary_context": None,
                "version": full_result.get("model_version", "1.0.0-legacy"),
            }

        safer_rewrite = full_result.get("safer_rewrite")
        flagged_words = full_result.get("flagged_words", [])

        agent_result = run_agent_pipeline(
            text=text,
            predictor_result=pred_res,
            safer_rewrite=safer_rewrite,
            flagged_words=flagged_words,
        )

        # Combine analysis result with agent pipeline result
        return jsonify({
            "analysis": full_result,
            "agent": agent_result,
        })

    except Exception as exc:
        return jsonify({"error": "Agent pipeline error: " + str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)