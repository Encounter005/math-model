import pytest

from scripts.question3.artifacts import write_attachment4_outputs
from scripts.question3.infer_attachment4 import ensemble_predictions


def test_ensemble_averages_probabilities_before_selecting_class():
    result = ensemble_predictions(
        [
            {"probabilities": [0.49, 0.51, 0.0], "regression_prediction": 0.0},
            {"probabilities": [0.9, 0.1, 0.0], "regression_prediction": 1.0},
        ]
    )
    assert result["class_prediction"] == 0
    assert result["regression_prediction"] == pytest.approx(0.5)


def test_attachment4_exports_prediction_and_evidence_csvs(tmp_path):
    write_attachment4_outputs(
        tmp_path,
        [{"id": "01", "class_prediction": 0}],
        [{"id": "01", "modality": "text"}],
    )
    assert (tmp_path / "attachment4_predictions.csv").is_file()
    assert (tmp_path / "attachment4_evidence.csv").is_file()
