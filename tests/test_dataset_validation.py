"""Unit tests for CyberShield AI Phase 1 Synthetic Dataset Validation.

Tests cover:
1. A valid record
2. Missing required fields
3. Duplicate IDs
4. Unknown labels
5. 'safe' combined with a harmful label
6. Empty comment text
7. Empty labels
8. Malformed dataset input
9. Multi-label records
10. Duplicate-comment detection (exact and near-duplicate)
11. Model feature vector mapping
12. Full validation of the actual data/synthetic_comments.json file
"""

import json
import os
import tempfile
import unittest

from validate_dataset import DatasetValidator, ORDERED_LABELS


class TestDatasetValidation(unittest.TestCase):
    """Test suite for dataset validation rules."""

    def setUp(self):
        self.validator = DatasetValidator(near_dup_threshold=0.85)

    # 1. Valid record
    def test_valid_record(self):
        record = {
            "id": "SYN001",
            "platform": "synthetic",
            "comment": "This is a polite and constructive comment.",
            "labels": ["safe"],
        }
        is_valid, errors, warnings = self.validator.validate_record(record, 0)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    # 2. Missing required fields
    def test_missing_required_fields(self):
        # Missing 'id'
        is_valid, errors, _ = self.validator.validate_record({"comment": "hello", "labels": ["safe"]}, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required field 'id'" in e for e in errors))

        # Missing 'comment'
        is_valid, errors, _ = self.validator.validate_record({"id": "SYN1", "labels": ["safe"]}, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required field 'comment'" in e for e in errors))

        # Missing 'labels'
        is_valid, errors, _ = self.validator.validate_record({"id": "SYN1", "comment": "hello"}, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required field 'labels'" in e for e in errors))

    # 3. Duplicate IDs
    def test_duplicate_ids(self):
        records = [
            {"id": "SYN001", "comment": "Comment A", "labels": ["safe"]},
            {"id": "SYN001", "comment": "Comment B", "labels": ["toxic"]},
        ]
        stats = self.validator.validate_dataset(records)
        self.assertEqual(stats["duplicate_id_count"], 1)
        self.assertTrue(any("Duplicate ID 'SYN001'" in e for e in stats["errors"]))

    # 4. Unknown labels
    def test_unknown_labels(self):
        record = {
            "id": "SYN001",
            "comment": "Some message",
            "labels": ["safe", "racist", "spam"],
        }
        is_valid, errors, _ = self.validator.validate_record(record, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("Unknown label 'racist'" in e for e in errors))
        self.assertTrue(any("Unknown label 'spam'" in e for e in errors))

    # 5. 'safe' combined with a harmful label
    def test_safe_combined_with_harmful_label(self):
        # safe + toxic
        record = {
            "id": "SYN001",
            "comment": "Mixed signals comment",
            "labels": ["safe", "toxic"],
        }
        is_valid, errors, _ = self.validator.validate_record(record, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("Label consistency violation" in e for e in errors))

        # safe + threat
        record_threat = {
            "id": "SYN002",
            "comment": "Nice day but watch your back",
            "labels": ["safe", "threat"],
        }
        is_valid, errors, _ = self.validator.validate_record(record_threat, 1)
        self.assertFalse(is_valid)
        self.assertTrue(any("Label consistency violation" in e for e in errors))

        # safe + insult
        record_insult = {
            "id": "SYN003",
            "comment": "Hello you idiot",
            "labels": ["safe", "insult"],
        }
        is_valid, errors, _ = self.validator.validate_record(record_insult, 2)
        self.assertFalse(is_valid)
        self.assertTrue(any("Label consistency violation" in e for e in errors))

        # safe + obscene
        record_obscene = {
            "id": "SYN004",
            "comment": "Good job bullshit",
            "labels": ["safe", "obscene"],
        }
        is_valid, errors, _ = self.validator.validate_record(record_obscene, 3)
        self.assertFalse(is_valid)
        self.assertTrue(any("Label consistency violation" in e for e in errors))

        # safe + sarcasm
        record_sarcasm = {
            "id": "SYN005",
            "comment": "Sure you did great",
            "labels": ["safe", "sarcasm"],
        }
        is_valid, errors, _ = self.validator.validate_record(record_sarcasm, 4)
        self.assertFalse(is_valid)
        self.assertTrue(any("cannot coexist with 'sarcasm'" in e for e in errors))

    # 6. Empty comment text
    def test_empty_comment_text(self):
        record_empty = {"id": "SYN001", "comment": "", "labels": ["safe"]}
        is_valid, errors, _ = self.validator.validate_record(record_empty, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("'comment' text is empty" in e for e in errors))

        record_whitespace = {"id": "SYN002", "comment": "   \n\t  ", "labels": ["safe"]}
        is_valid, errors, _ = self.validator.validate_record(record_whitespace, 1)
        self.assertFalse(is_valid)
        self.assertTrue(any("'comment' text is empty" in e for e in errors))

    # 7. Empty labels
    def test_empty_labels(self):
        record = {"id": "SYN001", "comment": "Nice post", "labels": []}
        is_valid, errors, _ = self.validator.validate_record(record, 0)
        self.assertFalse(is_valid)
        self.assertTrue(any("'labels' array is empty" in e for e in errors))

    # 8. Malformed dataset input
    def test_malformed_dataset_input(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            f.write("{ broken json syntax [")
            temp_path = f.name

        try:
            records, errors = self.validator.load_dataset(temp_path)
            self.assertTrue(len(errors) > 0)
            self.assertTrue(any("JSON parsing error" in e for e in errors))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Test root is not a list
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            f.write(json.dumps("just a string"))
            temp_path2 = f.name

        try:
            records, errors = self.validator.load_dataset(temp_path2)
            self.assertTrue(len(errors) > 0)
        finally:
            if os.path.exists(temp_path2):
                os.remove(temp_path2)

    # 9. Multi-label records
    def test_multi_label_records(self):
        records = [
            {
                "id": "SYN001",
                "comment": "Shut up you absolute clown.",
                "labels": ["toxic", "insult"],
            },
            {
                "id": "SYN002",
                "comment": "Oh look at the certified genius here.",
                "labels": ["toxic", "insult", "sarcasm"],
            },
        ]
        stats = self.validator.validate_dataset(records)
        self.assertEqual(stats["valid_records"], 2)
        self.assertEqual(stats["invalid_records"], 0)
        self.assertEqual(stats["multi_label_count"], 2)
        self.assertEqual(stats["label_counts"]["toxic"], 2)
        self.assertEqual(stats["label_counts"]["insult"], 2)
        self.assertEqual(stats["label_counts"]["sarcasm"], 1)

    # 10. Duplicate-comment detection
    def test_duplicate_comment_detection(self):
        # Exact duplicate
        records_exact = [
            {"id": "SYN001", "comment": "Hello world today", "labels": ["safe"]},
            {"id": "SYN002", "comment": "hello world today!", "labels": ["safe"]},
        ]
        stats_exact = self.validator.validate_dataset(records_exact)
        self.assertEqual(stats_exact["exact_duplicate_count"], 1)
        self.assertTrue(any("Exact duplicate comment detected" in e for e in stats_exact["errors"]))

        # Near-duplicate warning
        records_near = [
            {
                "id": "SYN001",
                "comment": "You did a truly fantastic job explaining that complex topic in our session today.",
                "labels": ["safe"],
            },
            {
                "id": "SYN002",
                "comment": "You did a truly fantastic job explaining that complex topic in our session.",
                "labels": ["safe"],
            },
        ]
        stats_near = self.validator.validate_dataset(records_near)
        self.assertTrue(len(stats_near["near_duplicates"]) > 0)
        self.assertTrue(any("Potential near-duplicate detected" in w for w in stats_near["warnings"]))

    # 11. Model feature vector mapping
    def test_model_vector_mapping(self):
        records = [
            {"id": "SYN001", "comment": "Good job", "labels": ["safe"]},
            {"id": "SYN002", "comment": "You idiot", "labels": ["toxic", "insult"]},
        ]
        mapped = self.validator.to_model_matrix(records)
        self.assertEqual(len(mapped), 2)
        # Order: safe, toxic, insult, threat, obscene, sarcasm
        self.assertEqual(mapped[0]["vector"], [1, 0, 0, 0, 0, 0])
        self.assertEqual(mapped[1]["vector"], [0, 1, 1, 0, 0, 0])

    # 12. Full validation of actual synthetic comments dataset file
    def test_actual_dataset_file(self):
        dataset_path = os.path.join("data", "synthetic_comments.json")
        self.assertTrue(os.path.exists(dataset_path), f"Dataset file missing at {dataset_path}")

        records, load_errors = self.validator.load_dataset(dataset_path)
        self.assertEqual(len(load_errors), 0, f"Load errors: {load_errors}")
        self.assertEqual(len(records), 100, f"Expected 100 records, got {len(records)}")

        stats = self.validator.validate_dataset(records)
        self.assertEqual(stats["invalid_records"], 0, f"Validation errors: {stats['errors']}")
        self.assertEqual(stats["valid_records"], 100)
        self.assertEqual(stats["unique_ids"], 100)
        self.assertEqual(stats["exact_duplicate_count"], 0)

        # Verify all 6 labels are present
        for lbl in ORDERED_LABELS:
            self.assertGreater(stats["label_counts"][lbl], 0, f"Label {lbl} has 0 occurrences")


if __name__ == "__main__":
    unittest.main()
