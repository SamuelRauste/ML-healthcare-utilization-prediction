"""Small, data-independent checks for the documented preprocessing pipeline."""

import unittest

import pandas as pd

from train import normalize_column_names, prepare_data


class TestPreprocessing(unittest.TestCase):
    def test_normalize_variants_and_typos(self):
        self.assertEqual(
            normalize_column_names(["Phyiscal Health", "Uknown Keeps Patient from Sleeping"]),
            ["Physical_Health", "Unknown_Keeps_Patient_from_Sleeping"],
        )

    def test_removes_refused_records_and_unused_features(self):
        df = pd.DataFrame(
            {
                "Number of Doctors Visited": [1, 2, 3],
                "Age": [1, 2, 2],
                "Phyiscal Health": [2, 3, -1],
                "Mental Health": [2, 2, 3],
                "Dental Health": [2, 3, 2],
                "Employment": [1, 3, 2],
                "Prescription Sleep Medication": [3, 3, 3],
                "Race": [1, 2, 1],
                "Trouble Sleeping": [0, 1, 0],
                "Gender": [1, 2, 1],
            }
        )
        X, y = prepare_data(df)
        self.assertEqual(len(X), 2)
        self.assertEqual(list(y), [1, 2])
        self.assertNotIn("Race", X.columns)
        self.assertNotIn("Trouble_Sleeping", X.columns)
        self.assertIn("Physical_Health", X.columns)


if __name__ == "__main__":
    unittest.main()
