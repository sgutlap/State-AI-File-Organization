from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from organize import MODEL_NAMES, organize_folder, prepare_training_data, train_model


def training_data() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "filename": f"assignment_{index}.pdf",
                "extension": ".pdf",
                "size_bytes": 100_000 + index * 100,
                "created_time": 1_700_000_000 + index,
                "modified_time": 1_710_000_000 + index,
                "accessed_time": 1_720_000_000 + index,
                "label": "wanted",
            }
        )
        rows.append(
            {
                "filename": f"cache_{index}.tmp",
                "extension": ".tmp",
                "size_bytes": 100 + index,
                "created_time": 1_600_000_000 + index,
                "modified_time": 1_600_000_100 + index,
                "accessed_time": 1_600_000_200 + index,
                "label": "unwanted",
            }
        )
    return pd.DataFrame(rows)


class SizeModel:
    def predict(self, features: pd.DataFrame):
        return [0 if int(features.iloc[0]["size_bytes"]) < 10 else 1]

    def predict_proba(self, features: pd.DataFrame):
        prediction = self.predict(features)[0]
        return [[0.9, 0.1] if prediction == 0 else [0.1, 0.9]]


class OrganizerTests(unittest.TestCase):
    def test_all_six_models_train_and_predict(self):
        data = training_data()
        features, _ = prepare_training_data(data)
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                model = train_model(data, model_name)
                prediction = int(model.predict(features.iloc[[0]])[0])
                self.assertIn(prediction, (0, 1))

    def test_dry_run_does_not_move_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "small.txt"
            source.write_text("small", encoding="utf-8")

            moves = organize_folder(folder, SizeModel(), apply=False)

            self.assertEqual(len(moves), 1)
            self.assertTrue(source.exists())
            self.assertFalse((folder / "Wanted").exists())
            self.assertFalse((folder / "Unwanted").exists())

    def test_apply_moves_files_without_overwriting_and_skips_expected_items(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            wanted = folder / "small.txt"
            unwanted = folder / "large.bin"
            hidden = folder / ".hidden.txt"
            training_csv = folder / "labels.csv"
            nested = folder / "nested"
            nested.mkdir()
            nested_file = nested / "inside.txt"
            wanted.write_text("small", encoding="utf-8")
            unwanted.write_bytes(b"x" * 20)
            hidden.write_text("hidden", encoding="utf-8")
            training_csv.write_text("training", encoding="utf-8")
            nested_file.write_text("nested", encoding="utf-8")
            (folder / "Wanted").mkdir()
            existing = folder / "Wanted" / "small.txt"
            existing.write_text("existing", encoding="utf-8")

            moves = organize_folder(
                folder,
                SizeModel(),
                apply=True,
                skip_paths=[training_csv],
            )

            self.assertEqual(len(moves), 2)
            self.assertEqual(existing.read_text(encoding="utf-8"), "existing")
            self.assertTrue((folder / "Wanted" / "small_1.txt").exists())
            self.assertTrue((folder / "Unwanted" / "large.bin").exists())
            self.assertTrue(hidden.exists())
            self.assertTrue(training_csv.exists())
            self.assertTrue(nested_file.exists())


if __name__ == "__main__":
    unittest.main()
