# CyberShield AI 🛡️

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**CyberShield AI** is a real-time, multi-label cyberbullying detection and content moderation platform. It combines machine learning, safety-first escalation policies, autonomous agentic moderation pipelines, explainable AI, and interactive analytics to identify harassment, threats, insults, and sarcasm online.

---

## 🌟 Key Capabilities

- **6-Label Multi-Label Schema**: Predicts probabilities independently across six distinct labels:
  1. `safe` — Constructive, polite, or harmless discourse.
  2. `toxic` — Rude, hostile, or disrespectful language.
  3. `insult` — Demeaning, mocking, or degrading attacks.
  4. `threat` — Physical harm, conditional violence, or stalking threats.
  5. `obscene` — Vulgar or profane language.
  6. `sarcasm` — Passive-aggressive, ironic, or mocking tone.
- **Sarcasm vs. Toxicity Disentanglement**: Sarcasm is modeled independently from toxicity (`sarcasm != toxic`). Benign irony (e.g. *"I love waiting an hour for a five-minute meeting"*) is correctly flagged as low severity without triggering false toxicity punishments.
- **Safety-First Threat Policy**: Threat recall threshold is tuned to `0.35`. Potential threats are escalated for human moderation review (`FLAG_FOR_HUMAN_REVIEW`) and never trigger automatic punitive bans.
- **Auxiliary Context & Idiom Guardrails**: Contextual idioms containing violent words (e.g., *"kill time"*, *"battery is dying"*, *"died laughing"*) are recognized as non-threatening context.
- **Dual-Metric Clarity**:
  - **Label confidence**: The probability of the predicted category (e.g. `90%` for sarcasm).
  - **Harmfulness score**: The highest probability among harmful categories (`threat`, `toxic`, `insult`, `obscene`), displaying `0%` for safe comments and residual toxicity (1%–3%) for benign sarcasm.
- **Inconclusive / Low-Confidence Handling**: When no category crosses threshold and safe probability is low, the system assigns `uncertain` with an informational neutral badge (`ℹ️ Inconclusive / Low Confidence`), ensuring zero false claims of confidence.
- **Agentic Moderation Pipeline (Phase 3)**: A 4-stage autonomous pipeline executing deep content analysis, risk appraisal, policy enforcement, and audit reasoning.
- **Word-Level Attribution & Safer Rewrites**: Highlights flagged lexical tokens and provides polite, constructive alternative rewrites.
- **Zero Binary Dependency**: Fast, pure Python inference using sublinear TF-IDF + Sigmoid logits with zero external binary or C++ runtime dependencies.
- **Zero-Downtime Rollback**: Instant rollback to the legacy V1 model via the `CYBERSHIELD_MODEL_VERSION=v1` environment variable.

---

## 🏗️ Project Architecture

```
cybershield-ai/
├── app.py                      # Flask web application & REST API routes
├── predictor.py                # High-level inference engine & decision policies
├── agent.py                    # 4-stage autonomous agentic moderation pipeline
├── train_multi_label.py        # Reproducible training pipeline for Model V2
├── validate_dataset.py         # Dataset schema, label balance, and matrix validator
├── build_dataset_v2.py         # Dataset builder & synthesizer for Model V2
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── synthetic_comments_v2.json  # Multi-label dataset (V2 training data)
│   ├── synthetic_comments.json     # Baseline dataset (V1)
│   ├── history.json                # Persisted moderation activity log
│   └── dataset_quality_report_v2.md# Dataset quality & balance report
│
├── model/
│   ├── cybershield_v2.json     # Trained Model V2 artifact (vocab, weights, biases)
│   └── cybershield.json        # Preserved Legacy Model V1 artifact
│
├── static/
│   ├── css/style.css           # Premium dark theme and responsive layout
│   └── js/
│       ├── analyzer.js         # Interactive analyzer frontend & agent visualization
│       └── dashboard.js        # Analytics charts and activity table
│
├── templates/
│   ├── base.html               # Shared navigation, styling, and layout shell
│   ├── index.html              # Landing page and platform features
│   ├── analyzer.html           # Live comment analyzer & agent pipeline UI
│   └── dashboard.html          # Analytics dashboard & metrics overview
│
└── tests/
    ├── test_dataset_validation.py  # Dataset schema & vector mapping tests (12 tests)
    └── test_multi_label_model.py   # Predictor, Flask API, and regression tests (26 tests)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/pradnya-182006/cybershield-ai.git
cd cybershield-ai
```

### 3. Create a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
pip install pytest
```

### 5. Start the Application
```bash
python app.py
```

The application will start on **`http://127.0.0.1:5000`**.

- **Home**: `http://127.0.0.1:5000/`
- **Text Analyzer**: `http://127.0.0.1:5000/analyzer`
- **Analytics Dashboard**: `http://127.0.0.1:5000/dashboard`

---

## 📡 API Reference

### 1. Analyze Text
**Endpoint:** `POST /api/analyze`

**Request:**
```json
{
  "text": "I absolutely love waiting an hour for a five-minute meeting."
}
```

**Response:**
```json
{
  "label": "sarcasm",
  "label_confidence": 0.9018,
  "harmfulness_score": 0.013,
  "severity": "low",
  "confidence": 0.8,
  "uncertain": false,
  "predicted_labels": ["sarcasm"],
  "scores": {
    "safe": 0.0539,
    "toxic": 0.0126,
    "insult": 0.0088,
    "threat": 0.0049,
    "obscene": 0.0079,
    "sarcasm": 0.9018
  },
  "explanation": "Contextual sarcasm detected without active toxicity. Sarcasm is context-dependent and reviewed separately.",
  "moderation_policy": {
    "action": "ALLOW_WITH_CONTEXT",
    "priority": "LOW",
    "reason": "Sarcastic language detected without active toxicity."
  },
  "safer_rewrite": "I have concerns regarding this matter and would like to address them directly and constructively.",
  "tone_tags": ["Sarcastic"],
  "model_version": "2.0.0"
}
```

### 2. Autonomous Agent Pipeline
**Endpoint:** `POST /api/agent`

**Request:**
```json
{
  "text": "You are an absolute idiot and I will find your street."
}
```

Runs the 4-stage pipeline:
1. `Content Analysis`: Ingests multi-label probabilities and lexical signals.
2. `Context & Threat Review`: Evaluates conditional threats, idioms, and target specificity.
3. `Policy Decision`: Executes resolution logic and sets action priority (`FLAG_FOR_HUMAN_REVIEW`).
4. `Explanation & Action Plan`: Generates human-readable rationale and suggested interventions.

### 3. Historical Analytics & Logs
- `GET /api/stats` — Summary metrics (total, harmful %, safe %, severity distribution, 7-day trend).
- `GET /api/history` — Returns the 50 most recent moderation records.
- `DELETE /api/history` — Clears activity logs.

---

## 🧪 Testing

The repository includes a comprehensive 38-test automated suite covering dataset integrity, inference consistency, threat detection, sarcasm separation, uncertainty preservation, and Flask endpoints.

Run the test suite:
```bash
python -m pytest -v
```

Expected output:
```text
============================= test session starts =============================
collected 38 items

tests/test_dataset_validation.py (12 passed)
tests/test_multi_label_model.py (26 passed)

============================= 38 passed in 13.5s ==============================
```

---

## 🔄 Rollback Mechanism

CyberShield AI supports zero-downtime rollback to the legacy V1 model:

```bash
# Run with legacy V1 model
CYBERSHIELD_MODEL_VERSION=v1 python app.py
```

The system automatically switches weight matrices, label spaces, and feature extraction to the legacy Kaggle-compatible schema without throwing dictionary or index exceptions.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
