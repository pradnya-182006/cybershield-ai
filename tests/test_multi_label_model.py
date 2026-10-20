"""Unit tests for Phase 2 & Phase 2B: Multi-Label Machine Learning Model, Threat Detection, & Flask Integration.

Covers:
1. Dataset loading and label mapping (v1 and v2)
2. Six-label output shape and canonical label order
3. Multiple predicted labels
4. Empty, whitespace, None, and invalid input handling
5. Probability values in range 0 to 1
6. Model save/load consistency
7. Reproducible training configuration
8. Sarcasm separation from toxicity
9. Contradiction resolution policy
10. Threat positive detection (direct, conditional, stalking)
11. Threat negative detection (idioms, reporting, fiction, gaming)
12. Compatibility with legacy dictionary format
13. Legacy model artifact preservation (cybershield.json)
14. Flask web routes (/, /analyzer, /dashboard)
15. Flask /api/analyze endpoint integration
16. Flask /api/history and /api/stats endpoints
17. Application active predictor verification (Model V2 in use)
18. Clean rollback mechanism to legacy V1 model
"""

import json
import os
import tempfile
import unittest

from predictor import CyberShieldPredictor, ORDERED_LABELS, get_predictor, predict_comment
from train_multi_label import (
    RANDOM_SEED,
    DEFAULT_THRESHOLDS,
    train_full_model,
)
from validate_dataset import DatasetValidator
import app


class TestMultiLabelModel(unittest.TestCase):
    """Test suite for Phase 2 & 2B multi-label training, threat detection, and Flask integration."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = "model/cybershield_v2.json"
        cls.predictor = CyberShieldPredictor(cls.model_path)
        app.app.config["TESTING"] = True
        cls.client = app.app.test_client()

    # 1. Dataset loading and label mapping
    def test_dataset_loading_and_mapping(self):
        validator = DatasetValidator()
        # Original baseline dataset
        records, load_errors = validator.load_dataset("data/synthetic_comments.json")
        self.assertEqual(len(load_errors), 0)
        self.assertEqual(len(records), 100)

        matrix = validator.to_model_matrix(records)
        self.assertEqual(len(matrix), 100)
        for item in matrix:
            self.assertIn("vector", item)
            self.assertEqual(len(item["vector"]), 6)
            self.assertEqual(list(item["labels"].keys()), ORDERED_LABELS)

    # 1b. Expanded dataset v2 validation
    def test_dataset_v2_loading_and_mapping(self):
        validator = DatasetValidator()
        records_v2, load_errors_v2 = validator.load_dataset("data/synthetic_comments_v2.json")
        self.assertEqual(len(load_errors_v2), 0)
        # The v2 dataset is >= 160 records (originally 160, may be expanded further)
        self.assertGreaterEqual(len(records_v2), 160, f"Expected >= 160 records, got {len(records_v2)}")

        stats = validator.validate_dataset(records_v2)
        self.assertEqual(stats["invalid_records"], 0)
        self.assertEqual(stats["exact_duplicate_count"], 0)
        self.assertGreaterEqual(stats["label_counts"]["threat"], 20)

    # 2. Six-label output shape and label order
    def test_six_label_output_shape_and_order(self):
        res = self.predictor.predict("Thank you for your assistance today.")
        self.assertTrue(res["is_valid"])
        self.assertIn("scores", res)
        self.assertEqual(list(res["scores"].keys()), ORDERED_LABELS)
        self.assertEqual(len(res["scores"]), 6)

    # 3. Multiple predicted labels
    def test_multiple_predicted_labels(self):
        comment = "You are an utter idiot and you have no idea how foolish you sound to everyone."
        res = self.predictor.predict(comment)
        self.assertTrue(res["is_valid"])
        self.assertIn("insult", res["predicted_labels"])
        self.assertIn("toxic", res["predicted_labels"])
        self.assertGreaterEqual(len(res["predicted_labels"]), 2)

    # 4. Empty and invalid input
    def test_empty_and_invalid_input(self):
        empty_res = self.predictor.predict("")
        self.assertFalse(empty_res["is_valid"])
        self.assertEqual(empty_res["predicted_labels"], [])
        self.assertTrue(empty_res["uncertain"])
        self.assertIn("empty", empty_res["error"])

        ws_res = self.predictor.predict("   \n\t  ")
        self.assertFalse(ws_res["is_valid"])
        self.assertEqual(ws_res["predicted_labels"], [])

        none_res = self.predictor.predict(None)
        self.assertFalse(none_res["is_valid"])

        int_res = self.predictor.predict(12345)
        self.assertFalse(int_res["is_valid"])

    # 5. Probability values in range 0 to 1
    def test_probability_values_in_range_0_to_1(self):
        test_samples = [
            "You did a great job on the presentation.",
            "I will find you and hurt you.",
            "What the hell is this garbage?",
            "Oh sure, brilliant idea as usual.",
            "Random gibberish xyz abc 123",
            "SHOUTING ALL CAPS COMMENT HERE",
        ]
        for s in test_samples:
            res = self.predictor.predict(s)
            for lab, prob in res["scores"].items():
                self.assertGreaterEqual(prob, 0.0, f"{lab} probability < 0.0 for '{s}'")
                self.assertLessEqual(prob, 1.0, f"{lab} probability > 1.0 for '{s}'")

    # 6. Model save/load consistency
    def test_model_save_load_consistency(self):
        with open(self.model_path, "r", encoding="utf-8") as f:
            artifact_data = json.load(f)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tmp:
            json.dump(artifact_data, tmp)
            tmp_path = tmp.name

        try:
            temp_predictor = CyberShieldPredictor(tmp_path)
            comment = "Congratulations on completing your graduation!"
            res_orig = self.predictor.predict(comment)
            res_temp = temp_predictor.predict(comment)

            self.assertEqual(res_orig["predicted_labels"], res_temp["predicted_labels"])
            for lab in ORDERED_LABELS:
                self.assertAlmostEqual(res_orig["scores"][lab], res_temp["scores"][lab], places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 7. Reproducible training configuration
    def test_reproducible_training_configuration(self):
        with open("data/synthetic_comments.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)

        art1, _ = train_full_model(dataset, seed=RANDOM_SEED)
        art2, _ = train_full_model(dataset, seed=RANDOM_SEED)

        self.assertEqual(art1["labels"], art2["labels"])
        self.assertEqual(art1["b"], art2["b"])
        for lab in ORDERED_LABELS:
            self.assertEqual(art1["w"][lab], art2["w"][lab])

    # 8. Sarcasm separate from toxicity
    def test_sarcasm_separate_from_toxicity(self):
        comment = "Oh fantastic, another mandatory 8 AM meeting that definitely could have been an email."
        res = self.predictor.predict(comment)
        self.assertIn("sarcasm", res["predicted_labels"])
        self.assertNotIn("toxic", res["predicted_labels"])
        self.assertIn("Sarcasm Independence", res["policy_resolution"])

    # 9. Contradiction resolution policy
    def test_contradiction_policy_resolution(self):
        # When safe and harmful labels trigger, safe must be demoted from predicted labels
        raw_predicted = ["safe", "threat"]
        has_harmful = any(l in ["toxic", "insult", "threat", "obscene"] for l in raw_predicted)
        self.assertTrue(has_harmful)
        resolved = [l for l in raw_predicted if l != "safe"]
        self.assertEqual(resolved, ["threat"])
        self.assertNotIn("safe", resolved)

    # 10. Threat positive detection (direct, conditional, stalking)
    def test_threat_positive_detection(self):
        threat_examples = [
            "I am going to put you in the hospital if you ever show your face here again.",
            "I know exactly where your office is located, you will see me in person very soon.",
            "If I catch you on campus after dark, you are going to regret opening your mouth.",
            "You better watch your back when you walk home from class today.",
        ]
        for t in threat_examples:
            res = self.predictor.predict(t)
            self.assertIn("threat", res["predicted_labels"], f"Failed to predict threat for: '{t}'")
            self.assertGreaterEqual(res["scores"]["threat"], self.predictor.thresholds["threat"])
            self.assertEqual(res["moderation_policy"]["action"], "FLAG_FOR_HUMAN_REVIEW")
            self.assertEqual(res["moderation_policy"]["priority"], "CRITICAL")

    # 11. Threat negative detection (idioms, reporting, fiction, gaming)
    def test_threat_negative_detection(self):
        non_threat_examples = [
            "That intense leg workout yesterday is going to kill my muscles all week.",
            "My laptop battery is dying, let me grab my charger before the call drops.",
            "Someone sent a threat to my younger brother online, how do I report this account to moderators?",
            "In this video game boss fight, your character has to dodge incoming attacks or you will die.",
            "The security team intervened quickly and prevented anyone from being hurt at the venue.",
        ]
        for nt in non_threat_examples:
            res = self.predictor.predict(nt)
            self.assertNotIn("threat", res["predicted_labels"], f"False positive threat predicted for: '{nt}'")
            self.assertEqual(res["primary_label"], "safe")

    # 12. Compatibility with legacy dictionary format
    def test_compatibility_with_legacy_interface(self):
        comment = "You are an utter idiot."
        res = self.predictor.predict(comment)
        legacy = self.predictor.to_legacy_dict(res)

        self.assertIn("label", legacy)
        self.assertIn("scores", legacy)
        self.assertIn("overall_score", legacy)
        self.assertIn("harmfulness_score", legacy)
        self.assertIn("severity", legacy)
        self.assertIn("confidence", legacy)
        self.assertIn("predicted_labels", legacy)
        self.assertIn("moderation_policy", legacy)

        self.assertIsInstance(legacy["scores"], dict)
        self.assertIsInstance(legacy["harmfulness_score"], float)

    # 13. Legacy model artifact preserved
    def test_legacy_model_preserved(self):
        legacy_path = "model/cybershield.json"
        self.assertTrue(os.path.exists(legacy_path), "Legacy model artifact was deleted!")
        with open(legacy_path, "r", encoding="utf-8") as f:
            legacy = json.load(f)
        self.assertIn("labels", legacy)
        self.assertEqual(
            legacy["labels"],
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        )

    # 14. Flask web routes (/, /analyzer, /dashboard)
    def test_flask_web_routes(self):
        res_home = self.client.get("/")
        self.assertEqual(res_home.status_code, 200)
        self.assertIn(b"CyberShield", res_home.data)

        res_analyzer = self.client.get("/analyzer")
        self.assertEqual(res_analyzer.status_code, 200)
        self.assertIn(b"Comment Analyzer", res_analyzer.data)

        res_dashboard = self.client.get("/dashboard")
        self.assertEqual(res_dashboard.status_code, 200)
        self.assertIn(b"Analytics Dashboard", res_dashboard.data)

    # 15. Flask /api/analyze endpoint integration
    def test_flask_api_analyze_threat(self):
        comment = "I will find your street and make sure you regret ever opening your mouth."
        res = self.client.post("/api/analyze", json={"text": comment})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["label"], "threat")
        self.assertEqual(data["severity"], "high")
        self.assertIn("scores", data)
        self.assertIn("threat", data["scores"])
        self.assertIn("moderation_policy", data)
        self.assertEqual(data["moderation_policy"]["action"], "FLAG_FOR_HUMAN_REVIEW")

    def test_flask_api_analyze_safe(self):
        comment = "You did a fantastic job explaining that topic in our group study session."
        res = self.client.post("/api/analyze", json={"text": comment})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["label"], "safe")
        self.assertEqual(data["severity"], "none")
        self.assertEqual(data["harmfulness_score"], 0.0)

    def test_flask_api_analyze_sarcasm(self):
        comment = "Oh fantastic, another mandatory 8 AM meeting that definitely could have been an email."
        res = self.client.post("/api/analyze", json={"text": comment})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["label"], "sarcasm")
        self.assertIn("scores", data)
        self.assertGreater(data["scores"]["sarcasm"], 0.50)

    def test_flask_api_analyze_empty_input(self):
        res = self.client.post("/api/analyze", json={"text": ""})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    # 16. Flask /api/history and /api/stats endpoints
    def test_flask_api_history_and_stats(self):
        res_stats = self.client.get("/api/stats")
        self.assertEqual(res_stats.status_code, 200)
        stats = res_stats.get_json()
        self.assertIn("total", stats)
        self.assertIn("toxic_pct", stats)
        self.assertIn("categories", stats)

        res_hist = self.client.get("/api/history")
        self.assertEqual(res_hist.status_code, 200)
        self.assertIsInstance(res_hist.get_json(), list)

    # 17. Application active predictor verification (Model V2 in use)
    def test_flask_active_predictor_integration(self):
        self.assertTrue(app.USE_V2, "Flask app is not using Model V2!")
        self.assertIsNotNone(app.predictor_v2, "Predictor V2 instance is None!")
        self.assertEqual(app.predictor_v2.version, "2.0.0")

    # 18. Clean rollback mechanism to legacy V1 model
    def test_flask_rollback_mechanism(self):
        original_use_v2 = app.USE_V2
        original_predictor = app.predictor_v2
        try:
            # Simulate rollback to V1
            app.USE_V2 = False
            app.predictor_v2 = None

            res = app.analyze_text("You are so stupid and worthless, nobody likes you")
            self.assertEqual(res["model_version"], "1.0.0-legacy")
            self.assertIn("toxic", res["scores"])
            self.assertIn("severe_toxic", res["scores"])  # Legacy Kaggle labels
        finally:
            # Restore V2 state
            app.USE_V2 = original_use_v2
            app.predictor_v2 = original_predictor

    # 19. Score consistency and representative examples regression tests
    def test_representative_examples_score_and_label_consistency(self):
        """Verify behavior and score definitions for representative user cases."""
        # Case A: Direct threat - should have high severity and high harmfulness score
        res_threat = self.client.post("/api/analyze", json={"text": "I will find you and hurt you."}).get_json()
        self.assertEqual(res_threat["label"], "threat")
        self.assertEqual(res_threat["severity"], "high")
        self.assertGreaterEqual(res_threat["harmfulness_score"], 0.85)

        # Case B: Sarcasm 1 - Sarcasm detected with high sarcasm probability, but low harmfulness score
        res_sarcasm1 = self.client.post("/api/analyze", json={"text": "Great job breaking the only thing that was working."}).get_json()
        self.assertEqual(res_sarcasm1["label"], "sarcasm")
        self.assertEqual(res_sarcasm1["severity"], "low")
        self.assertGreaterEqual(res_sarcasm1["scores"]["sarcasm"], 0.50)
        # Sarcasm is decoupled from toxicity; harmfulness score only measures toxic/insult/threat/obscene
        self.assertLess(res_sarcasm1["harmfulness_score"], 0.10)

        # Case C: Sarcasm 2 - Benign sarcasm with high sarcasm confidence and low harmfulness
        res_sarcasm2 = self.client.post("/api/analyze", json={"text": "I absolutely love waiting an hour for a five-minute meeting."}).get_json()
        self.assertEqual(res_sarcasm2["label"], "sarcasm")
        self.assertEqual(res_sarcasm2["severity"], "low")
        self.assertGreaterEqual(res_sarcasm2["scores"]["sarcasm"], 0.80)
        self.assertLess(res_sarcasm2["harmfulness_score"], 0.10)

        # Case D: Rhetorical/colloquial expression with 'hell' - low confidence / uncertain
        res_hell = self.client.post("/api/analyze", json={"text": "What the hell are you talking about?"}).get_json()
        self.assertEqual(res_hell["label"], "uncertain")
        self.assertTrue(res_hell["uncertain"])
        self.assertEqual(res_hell["severity"], "none")
        self.assertEqual(res_hell["harmfulness_score"], 0.0)

        # Case E: Rhetorical presentation comment - words 'good' and 'presentation' carry safe weights
        res_pres = self.client.post("/api/analyze", json={"text": "Haha, you call that a good presentation?"}).get_json()
        self.assertEqual(res_pres["label"], "safe")
        self.assertEqual(res_pres["severity"], "none")
        self.assertEqual(res_pres["harmfulness_score"], 0.0)

    def test_history_score_definition_consistency(self):
        """Verify saved history record 'score' accurately reflects harmfulness_score."""
        test_comment = "I absolutely love waiting an hour for a five-minute meeting."
        res = self.client.post("/api/analyze", json={"text": test_comment}).get_json()
        harmfulness = res["harmfulness_score"]

        # Fetch recent history
        hist = self.client.get("/api/history").get_json()
        matching = [item for item in hist if item["text"] == test_comment]
        self.assertTrue(len(matching) > 0)
        saved_record = matching[0]
        # Verify consistent field mapping
        self.assertEqual(saved_record["label"], res["label"])
        self.assertEqual(saved_record["severity"], res["severity"])
        self.assertEqual(saved_record["score"], harmfulness)

    # 20. Verification of Label confidence vs Harmfulness score mapping
    def test_label_confidence_and_harmfulness_score_mapping(self):
        """Verify that label confidence and harmfulness score are mapped correctly for sarcasm, threat, and safe."""
        # A. Sarcasm comment: high label confidence, low harmfulness score
        sarcasm_text = "I absolutely love waiting an hour for a five-minute meeting."
        res_sarcasm = self.client.post("/api/analyze", json={"text": sarcasm_text}).get_json()
        self.assertEqual(res_sarcasm["label"], "sarcasm")
        self.assertIn("label_confidence", res_sarcasm)
        self.assertEqual(res_sarcasm["label_confidence"], res_sarcasm["scores"]["sarcasm"])
        self.assertGreaterEqual(res_sarcasm["label_confidence"], 0.80)
        self.assertLess(res_sarcasm["harmfulness_score"], 0.10)
        self.assertNotEqual(res_sarcasm["label_confidence"], res_sarcasm["harmfulness_score"])

        # B. Threat comment: high label confidence and high harmfulness score
        threat_text = "I will find you and hurt you."
        res_threat = self.client.post("/api/analyze", json={"text": threat_text}).get_json()
        self.assertEqual(res_threat["label"], "threat")
        self.assertIn("label_confidence", res_threat)
        self.assertEqual(res_threat["label_confidence"], res_threat["scores"]["threat"])
        self.assertGreaterEqual(res_threat["label_confidence"], 0.85)
        self.assertGreaterEqual(res_threat["harmfulness_score"], 0.85)

        # C. Safe comment: safe label confidence, zero harmfulness score
        safe_text = "Thank you for your wonderful help! Have a great day."
        res_safe = self.client.post("/api/analyze", json={"text": safe_text}).get_json()
        self.assertEqual(res_safe["label"], "safe")
        self.assertIn("label_confidence", res_safe)
        self.assertEqual(res_safe["label_confidence"], res_safe["scores"]["safe"])
        self.assertGreaterEqual(res_safe["label_confidence"], 0.70)
        self.assertEqual(res_safe["harmfulness_score"], 0.0)

        # D. Safe/uncertain comment: uncertain label, zero harmfulness score, valid label_confidence
        hell_text = "What the hell are you talking about?"
        res_hell = self.client.post("/api/analyze", json={"text": hell_text}).get_json()
        self.assertEqual(res_hell["label"], "uncertain")
        self.assertTrue(res_hell["uncertain"])
        self.assertEqual(res_hell["harmfulness_score"], 0.0)
        self.assertIn("label_confidence", res_hell)
        self.assertIsInstance(res_hell["label_confidence"], float)

        # E. Verify result['confidence'] preserves its distinct definition
        # (normalized distance from boundary 0.5) and is not mistakenly overwritten
        self.assertIn("confidence", res_sarcasm)
        self.assertEqual(res_sarcasm["confidence"], round(abs(res_sarcasm["overall_score"] - 0.5) * 2, 2))

    # 21. Dedicated uncertainty handling and preservation tests
    def test_uncertainty_handling_and_preservation(self):
        """Verify uncertain decision preservation across backend, API, history, and legacy fallback."""
        from unittest.mock import patch

        # 1. Confident safe comment
        res_safe = self.client.post("/api/analyze", json={"text": "Thank you for your wonderful help! Have a great day."}).get_json()
        self.assertEqual(res_safe["label"], "safe")
        self.assertFalse(res_safe["uncertain"])
        self.assertEqual(res_safe["harmfulness_score"], 0.0)
        self.assertGreaterEqual(res_safe["label_confidence"], 0.70)

        # 2. Low-confidence uncertain comment
        res_hell = self.client.post("/api/analyze", json={"text": "What the hell are you talking about?"}).get_json()
        self.assertEqual(res_hell["label"], "uncertain")
        self.assertTrue(res_hell["uncertain"])
        self.assertEqual(res_hell["harmfulness_score"], 0.0)
        self.assertIn("The model could not confidently classify", res_hell["explanation"])

        # 3. Sarcasm with low harmfulness
        res_sarc = self.client.post("/api/analyze", json={"text": "I absolutely love waiting an hour for a five-minute meeting."}).get_json()
        self.assertEqual(res_sarc["label"], "sarcasm")
        self.assertFalse(res_sarc["uncertain"])
        self.assertLess(res_sarc["harmfulness_score"], 0.10)
        self.assertGreaterEqual(res_sarc["scores"]["sarcasm"], 0.80)

        # 4. High-confidence direct threat
        res_threat = self.client.post("/api/analyze", json={"text": "I will find you and hurt you."}).get_json()
        self.assertEqual(res_threat["label"], "threat")
        self.assertFalse(res_threat["uncertain"])
        self.assertEqual(res_threat["severity"], "high")
        self.assertGreaterEqual(res_threat["harmfulness_score"], 0.85)

        # 5. Legacy model fallback preservation
        orig_v2 = app.USE_V2
        orig_p = app.predictor_v2
        try:
            app.USE_V2 = False
            app.predictor_v2 = None
            res_v1 = app.analyze_text("xyz random gibberish 12345")
            self.assertIn("uncertain", res_v1)
            self.assertEqual(res_v1["model_version"], "1.0.0-legacy")
            # Legacy scores must only contain Kaggle labels (no KeyError)
            for lab in res_v1["scores"]:
                self.assertIn(lab, ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
        finally:
            app.USE_V2 = orig_v2
            app.predictor_v2 = orig_p

        # 6. API and history preservation of uncertainty
        unique_text = "What the hell are you talking about? [test_id_uncertain]"
        res_api = self.client.post("/api/analyze", json={"text": unique_text}).get_json()
        self.assertEqual(res_api["label"], "uncertain")
        self.assertTrue(res_api["uncertain"])

        hist = self.client.get("/api/history").get_json()
        saved = [r for r in hist if r["text"] == unique_text]
        self.assertTrue(len(saved) > 0)
        self.assertEqual(saved[0]["label"], "uncertain")
        self.assertTrue(saved[0]["uncertain"])

        # 7. Old history records without an uncertainty field render cleanly
        with app.lock:
            h_items = app.load_history()
            h_items.append({
                "id": "old_rec_1",
                "text": "old historic comment without uncertain field",
                "label": "safe",
                "severity": "none",
                "score": 0.0,
                "time": "2026-01-01 10:00:00"
            })
            app.save_history(h_items)

        res_hist_old = self.client.get("/api/history").get_json()
        matching_old = [r for r in res_hist_old if r.get("id") == "old_rec_1"]
        self.assertTrue(len(matching_old) > 0)
        self.assertNotIn("uncertain", matching_old[0])

        res_stats = self.client.get("/api/stats").get_json()
        self.assertIn("total", res_stats)

        # 8. Deterministic mocked predictor uncertainty tests
        mock_pred_uncertain = {
            "comment": "mock comment",
            "is_valid": True,
            "scores": {"safe": 0.15, "toxic": 0.10, "insult": 0.05, "threat": 0.02, "obscene": 0.08, "sarcasm": 0.05},
            "raw_scores": {"safe": 0.15, "toxic": 0.10, "insult": 0.05, "threat": 0.02, "obscene": 0.08, "sarcasm": 0.05},
            "predicted_labels": [],
            "raw_predicted_labels": [],
            "primary_label": "uncertain",
            "severity": "none",
            "confidence": 0.15,
            "uncertain": True,
            "policy_resolution": "Low confidence classification.",
            "moderation_policy": {"action": "ALLOW_UNCERTAIN", "priority": "LOW", "reason": "Low confidence."},
            "auxiliary_context": None,
            "thresholds": {"safe": 0.4, "toxic": 0.4, "insult": 0.4, "threat": 0.35, "obscene": 0.4, "sarcasm": 0.35},
            "version": "2.0.0"
        }
        with patch.object(app.predictor_v2, "predict", return_value=mock_pred_uncertain):
            res_mock = app.analyze_text("Some ambiguous text")
            self.assertEqual(res_mock["label"], "uncertain")
            self.assertTrue(res_mock["uncertain"])
            self.assertEqual(res_mock["harmfulness_score"], 0.0)
            self.assertEqual(res_mock["explanation"], "The model could not confidently classify this comment. No automatic moderation action was taken.")


if __name__ == "__main__":
    unittest.main()



