# CyberShield AI — Synthetic Dataset Quality Report

**Dataset File:** `data/synthetic_comments.json`  
**Validation Status:** ✅ PASSED  
**Health Score:** 100.0% (100/100 valid records)

---

## 1. Executive Summary

| Metric | Value | Target | Status |
| :--- | :---: | :---: | :---: |
| **Total Comments** | 100 | 100 | ✅ On Target |
| **Valid Records** | 100 | 100 | ✅ All Valid |
| **Invalid Records** | 0 | 0 | ✅ 0 Errors |
| **Unique IDs** | 100 | 100 | ✅ Unique |
| **Exact Duplicates** | 0 | 0 | ✅ Clean |
| **Near Duplicates** | 0 | 0-3 | ✅ Clean |
| **Multi-Label Comments** | 18 | 15–25 | ✅ Rich Co-occurrence |

---

## 2. Six-Label Multi-Label Distribution

| Label | Count | % of Dataset | Approximate Target | Status |
| :--- | :---: | :---: | :---: | :--- |
| `safe` | 31 | 31.0% | ~25 | ✅ Calibrated |
| `toxic` | 26 | 26.0% | ~20 | ✅ Calibrated |
| `insult` | 18 | 18.0% | ~15 | ✅ Calibrated |
| `threat` | 8 | 8.0% | ~8 | ✅ Calibrated |
| `obscene` | 16 | 16.0% | ~12 | ✅ Calibrated |
| `sarcasm` | 23 | 23.0% | ~20 | ✅ Calibrated |

---

## 3. Co-occurrence & Multi-Label Breakdown

Total comments with multiple labels: **18**

| Label Combination | Frequency |
| :--- | :---: |
| `insult + toxic` | 6 |
| `insult + sarcasm + toxic` | 4 |
| `sarcasm + toxic` | 3 |
| `threat + toxic` | 3 |
| `obscene + toxic` | 2 |

---

## 4. Validation Errors & Issues

✅ **Zero validation errors detected.** All records strictly adhere to the schema.

---

## 5. Quality Warnings & Observations

✅ **No quality warnings.** Dataset balance and uniqueness meet standards.

---

## 6. Multi-Label Model Mapping Verification

All valid records can be seamlessly mapped into binary label vectors for training in Phase 2:

```python
# Binary vector schema: [safe, toxic, insult, threat, obscene, sarcasm]
# Example: {"id": "SYN001", "labels": ["safe"]} -> [1, 0, 0, 0, 0, 0]
# Example: {"id": "SYN036", "labels": ["toxic", "insult", "sarcasm"]} -> [0, 1, 1, 0, 0, 1]
```

Phase 1 validation is **complete and verified**.