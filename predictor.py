"""CyberShield AI - Reusable Multi-Label Prediction Interface (Phase 2B).

Provides high-level, production-ready inference for the 6-label multi-label
cyberbullying and toxicity classification schema:
    1. safe
    2. toxic
    3. insult
    4. threat
    5. obscene
    6. sarcasm

Key Safety & Policy Capabilities:
- Independent multi-label probability outputs (0.0 to 1.0).
- Calibrated threat decision threshold (0.35) tuned on validation data.
- Threat safety policy: flags potential threats for human moderation review;
  never executes automatic punitive actions.
- Sarcasm strictly separated from toxicity (sarcasm != toxic).
- Contradiction resolution: harmful labels take precedence over 'safe' in
  predicted labels, while raw probabilities are strictly preserved.
- Auxiliary context detection for non-threatening idioms (e.g. "kill time",
  "battery is dying", "died laughing") and reporting statements to prevent
  simplistic keyword false positives.
- Backward-compatible dictionary adapter for Flask app callers.
- Zero external runtime binary dependencies.
"""

import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

ORDERED_LABELS: List[str] = [
    "safe",
    "toxic",
    "insult",
    "threat",
    "obscene",
    "sarcasm",
]

HARMFUL_LABELS: Set[str] = {
    "toxic",
    "insult",
    "threat",
    "obscene",
}

DEFAULT_MODEL_PATH = "model/cybershield_v2.json"
LEGACY_MODEL_PATH = "model/cybershield.json"

# Common harmless idioms and reporting phrases that contain violent words
# Used as an auxiliary context cue (NOT a sole filter or bypass)
NON_THREAT_IDIOMS: Set[str] = {
    "kill time",
    "kill my",
    "killing me",
    "died laughing",
    "dying laughing",
    "battery is dying",
    "eat a horse",
    "report a threat",
    "report this threat",
    "prevented anyone from being hurt",
    "boss fight",
    "video game",
}


# ── Sarcasm lexical cues ─────────────────────────────────────────────────
# Positive-sentiment words used in obviously negative/frustrating contexts
_SARCASM_OPENER_WORDS: Set[str] = {
    "oh", "wow", "great", "fantastic", "wonderful", "brilliant", "amazing",
    "sure", "absolutely", "totally", "obviously", "clearly", "naturally",
    "right", "perfect", "charming", "delightful", "thrilling", "spectacular",
    "commendable", "inspiring", "kudos", "bravo", "congratulations",
    "superb", "splendid", "magnificent", "remarkable", "impressive",
    "refreshing", "lovely", "terrific", "truly",
}

# Negative-context anchor words — when close to mock-praise, signal sarcasm
_SARCASM_CONTEXT_WORDS: Set[str] = {
    "broken", "crashed", "delayed", "late", "wrong", "ruined", "missed",
    "failed", "useless", "pointless", "nothing", "nobody", "never",
    "worse", "worst", "outdated", "impossible", "wasted", "forgot",
    "forgot", "empty", "zero", "no", "not", "complaint", "stuck",
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

    # ── Sarcasm structural signals ─────────────────────────────────────────
    word_set = set(words)

    # 1. Mock-praise opener: sentence starts with a sarcastic positive word
    if words and words[0] in _SARCASM_OPENER_WORDS:
        signals.append("__sarcasm_opener__")

    # 2. Contrast co-occurrence: mock-praise word AND negative-context word in same text
    has_praise = bool(word_set & _SARCASM_OPENER_WORDS)
    has_negative = bool(word_set & _SARCASM_CONTEXT_WORDS)
    if has_praise and has_negative:
        signals.append("__sarcasm_contrast__")

    # 3. Rhetorical intensifier before negative word ('absolutely nothing', 'clearly wrong')
    intensifiers = {"absolutely", "obviously", "clearly", "totally", "definitely", "naturally", "truly"}
    for i, w in enumerate(words[:-1]):
        if w in intensifiers and words[i + 1] in _SARCASM_CONTEXT_WORDS:
            signals.append("__sarcasm_intensifier__")
            break

    # 4. Comma-separated ironic constructions ('oh great, the X broke')
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
    """Bounded sigmoid function."""
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class CyberShieldPredictor:
    """Thread-safe multi-label predictor for CyberShield AI."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.version = "unknown"
        self.labels: List[str] = list(ORDERED_LABELS)
        # Calibrated decision thresholds
        # sarcasm lowered to 0.35 (same as threat) to prioritize recall given limited training data.
        self.thresholds: Dict[str, float] = {
            "safe": 0.40,
            "toxic": 0.40,
            "insult": 0.40,
            "threat": 0.35,  # Lower threshold prioritizing safety recall
            "obscene": 0.40,
            "sarcasm": 0.35,  # Lower threshold: sarcasm is highly contextual, recall > precision
        }
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.w: Dict[str, List[float]] = {}
        self.b: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load and parse the JSON model artifact."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Run 'python train_multi_label.py' to generate the model artifact."
            )

        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.version = data.get("version", "2.0.0")
        self.labels = data.get("labels", ORDERED_LABELS)
        self.thresholds = data.get("thresholds", self.thresholds)
        self.vocab = data.get("vocab", {})
        self.idf = data.get("idf", [])
        self.w = data.get("w", {})
        self.b = data.get("b", {})
        self.metadata = data.get("metadata", {})

    def vectorize(self, features: List[str]) -> List[Tuple[int, float]]:
        """Compute normalized sublinear TF-IDF vector."""
        tf = Counter(x for x in features if x in self.vocab)
        if not tf:
            return []
        vec = {self.vocab[t]: (1.0 + math.log(c)) * self.idf[self.vocab[t]] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return [(i, v / norm) for i, v in vec.items()]

    def predict(self, comment: Any) -> Dict[str, Any]:
        """Perform multi-label prediction on input text with full policy evaluation."""
        # 1. Validation of input
        if not isinstance(comment, str) or not comment.strip():
            return {
                "comment": comment if isinstance(comment, str) else str(comment),
                "is_valid": False,
                "error": "Input comment is empty, whitespace-only, or invalid.",
                "scores": {lab: 0.0 for lab in self.labels},
                "raw_scores": {lab: 0.0 for lab in self.labels},
                "predicted_labels": [],
                "raw_predicted_labels": [],
                "primary_label": "none",
                "severity": "none",
                "confidence": 0.0,
                "uncertain": True,
                "policy_resolution": "Rejected invalid input.",
                "moderation_policy": {
                    "action": "IGNORE_INVALID",
                    "priority": "NONE",
                    "reason": "Empty or invalid input string.",
                },
                "auxiliary_context": None,
                "version": self.version,
            }

        text = comment.strip()
        feats = extract_features(text)
        vec = self.vectorize(feats)

        # 2. Raw inference
        scores: Dict[str, float] = {}
        for lab in self.labels:
            if lab in self.w and lab in self.b and vec:
                z = self.b[lab] + sum(self.w[lab][j] * val for j, val in vec)
                prob = sigmoid(z)
            elif lab in self.b:
                prob = sigmoid(self.b[lab])
            else:
                prob = 0.0
            scores[lab] = round(prob, 4)

        # 3. Determine threshold crossings
        raw_predicted: List[str] = [
            lab for lab in self.labels if scores[lab] >= self.thresholds.get(lab, 0.40)
        ]

        # 4. Uncertainty & Low-confidence detection
        max_score = max(scores.values()) if scores else 0.0
        uncertain = False
        if not vec:
            uncertain = True
        elif max_score < 0.30:
            uncertain = True

        # Check auxiliary context for idioms or reporting contexts
        text_lower = text.lower()
        auxiliary_context: Optional[str] = None
        for idiom in NON_THREAT_IDIOMS:
            if idiom in text_lower:
                auxiliary_context = f"Non-threat phrase detected ('{idiom}'). Auxiliary review recommended."
                break

        # 5. Contradiction Resolution Policy (Harmful-Precedence Resolution)
        resolved_predicted = list(raw_predicted)
        policy_notes: List[str] = []

        has_harmful = any(lab in HARMFUL_LABELS for lab in raw_predicted)

        if "safe" in raw_predicted and has_harmful:
            resolved_predicted = [lab for lab in resolved_predicted if lab != "safe"]
            harmful_found = [lab for lab in raw_predicted if lab in HARMFUL_LABELS]
            policy_notes.append(
                f"Contradiction Policy Applied: 'safe' (p={scores['safe']}) was demoted because "
                f"harmful label(s) crossed thresholds: {harmful_found}."
            )
        elif "safe" in raw_predicted and "sarcasm" in raw_predicted and not has_harmful:
            resolved_predicted = ["sarcasm"]
            policy_notes.append(
                f"Benign Sarcasm Detected: Safe tone accompanied by contextual sarcasm (p={scores['sarcasm']})."
            )
        elif not resolved_predicted:
            if max_score < 0.35:
                policy_notes.append("Low confidence: no label crossed decision threshold.")
            else:
                policy_notes.append("Borderline score: no label crossed threshold.")

        # 6. Sarcasm Independence Guarantee
        if "sarcasm" in resolved_predicted and "toxic" not in raw_predicted:
            policy_notes.append("Sarcasm Independence: Comment evaluated as non-toxic sarcasm.")

        # 7. Threat Moderation Policy & Prioritization
        threat_score = scores.get("threat", 0.0)
        threat_thresh = self.thresholds.get("threat", 0.35)

        if threat_score >= threat_thresh or "threat" in resolved_predicted:
            severity = "high"
            primary_label = "threat"
            moderation_policy = {
                "action": "FLAG_FOR_HUMAN_REVIEW",
                "priority": "CRITICAL",
                "reason": (
                    f"Threat probability ({threat_score:.2f}) exceeded threshold ({threat_thresh:.2f}). "
                    "Escalate immediately for human moderation review. Do not apply automatic punitive bans."
                ),
            }
        elif 0.20 <= threat_score < threat_thresh:
            harmful_resolved = [l for l in resolved_predicted if l in HARMFUL_LABELS]
            primary_label = max(harmful_resolved, key=lambda l: scores[l]) if harmful_resolved else ("safe" if "safe" in resolved_predicted else "uncertain")
            severity = "medium" if harmful_resolved else "low"
            moderation_policy = {
                "action": "ELEVATED_MONITORING",
                "priority": "ELEVATED",
                "reason": f"Borderline threat probability ({threat_score:.2f}). Logged for priority review.",
            }
        elif any(l in HARMFUL_LABELS for l in resolved_predicted):
            harmful_resolved = [l for l in resolved_predicted if l in HARMFUL_LABELS]
            top_harmful = max(harmful_resolved, key=lambda l: scores[l])
            primary_label = top_harmful
            severity = "medium" if scores[top_harmful] >= 0.65 else "low"
            moderation_policy = {
                "action": "SUGGEST_REWRITE_OR_WARN",
                "priority": "STANDARD",
                "reason": f"Flagged as '{top_harmful}' ({scores[top_harmful]:.2f}). Suggest polite revision.",
            }
        elif "sarcasm" in resolved_predicted:
            primary_label = "sarcasm"
            severity = "low"
            moderation_policy = {
                "action": "ALLOW_WITH_CONTEXT",
                "priority": "LOW",
                "reason": "Sarcastic language detected without active toxicity.",
            }
        elif "safe" in resolved_predicted:
            primary_label = "safe"
            severity = "none"
            moderation_policy = {
                "action": "ALLOW",
                "priority": "NONE",
                "reason": "Message classified as safe and constructive.",
            }
        else:
            primary_label = "safe" if scores.get("safe", 0) > 0.30 else "uncertain"
            severity = "none"
            moderation_policy = {
                "action": "ALLOW_UNCERTAIN",
                "priority": "LOW",
                "reason": "Low confidence classification; treated as safe with uncertainty notice.",
            }

        confidence = round(max_score, 4)

        return {
            "comment": text,
            "is_valid": True,
            "scores": scores,
            "raw_scores": dict(scores),
            "predicted_labels": resolved_predicted,
            "raw_predicted_labels": raw_predicted,
            "primary_label": primary_label,
            "severity": severity,
            "confidence": confidence,
            "uncertain": uncertain,
            "policy_resolution": " | ".join(policy_notes) if policy_notes else "Clean classification.",
            "moderation_policy": moderation_policy,
            "auxiliary_context": auxiliary_context,
            "thresholds": dict(self.thresholds),
            "version": self.version,
        }

    def to_legacy_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert multi-label result into dictionary compatible with existing app.py callers."""
        scores = result["scores"]
        label = result["primary_label"]
        harmful_labels = [l for l in result["predicted_labels"] if l in HARMFUL_LABELS]

        harmfulness_score = max([scores[l] for l in HARMFUL_LABELS]) if harmful_labels else 0.0

        return {
            "label": label,
            "scores": scores,
            "overall_score": result["confidence"],
            "harmfulness_score": round(harmfulness_score, 3),
            "severity": result["severity"],
            "confidence": result["confidence"],
            "flagged_words": [],
            "predicted_labels": result["predicted_labels"],
            "moderation_policy": result["moderation_policy"],
            "auxiliary_context": result.get("auxiliary_context"),
            "explanation": f"Classified as '{label}' with {result['severity']} severity.",
            "disclaimer": "Predictions are estimates. Context, sarcasm and culture can change meaning.",
            "version": result["version"],
        }


# Singleton accessor for efficient reuse
_global_predictor: Optional[CyberShieldPredictor] = None


def get_predictor(model_path: str = DEFAULT_MODEL_PATH) -> CyberShieldPredictor:
    """Get or create singleton instance of CyberShieldPredictor."""
    global _global_predictor
    if _global_predictor is None or _global_predictor.model_path != model_path:
        _global_predictor = CyberShieldPredictor(model_path)
    return _global_predictor


def predict_comment(comment: str, model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Convenience function for standalone prediction."""
    return get_predictor(model_path).predict(comment)
