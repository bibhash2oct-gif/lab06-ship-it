"""Tests for the checks, the decision and the loading in predict.py."""

import importlib
import json
from pathlib import Path

import joblib
import pytest

import predict

MODEL_INFO_PATH = Path(__file__).resolve().parents[1] / "models" / "model_info.json"


def test_probability_is_between_0_and_1(valid_order):
    result = predict.predict_order(valid_order)
    assert 0 <= result["probability"] <= 1


def test_harder_order_scores_higher(easy_order, hard_order):
    easy = predict.predict_order(easy_order)["probability"]
    hard = predict.predict_order(hard_order)["probability"]
    assert hard > easy


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.30, True),  # exactly at the line: offer the voucher
        (0.299, False),  # just below: no voucher
        (0.301, True),  # just above: offer the voucher
    ],
)
def test_decision_at_the_boundary(probability, expected):
    # worked out by hand with a threshold of 0.30
    assert predict.decide(probability, threshold=0.30) is expected


def test_decision_uses_the_saved_threshold(valid_order):
    saved = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))["threshold"]
    assert predict.THRESHOLD == saved
    assert predict.predict_order(valid_order)["threshold"] == saved
    assert predict.decide(saved) is True
    assert predict.decide(saved - 0.001) is False


@pytest.mark.parametrize("empty", [None, "", float("nan")])
def test_missing_prep_minutes_is_accepted_with_a_warning(valid_order, empty):
    valid_order["prep_minutes"] = empty
    result = predict.predict_order(valid_order)
    assert 0 <= result["probability"] <= 1
    assert "prep_minutes" in result["warnings"][0]


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("distance_km", 0),
        ("distance_km", 31),
        ("items", 0),
        ("items", 21),
        ("items", 2.5),
        ("hour", 9),
        ("hour", 23),
        ("prep_minutes", 0),
        ("prep_minutes", 61),
        ("courier_months", -1),
        ("courier_months", 121),
        ("distance_km", "4.2"),
    ],
)
def test_impossible_number_is_rejected(valid_order, column, bad_value):
    valid_order[column] = bad_value
    with pytest.raises(ValueError, match=column):
        predict.predict_order(valid_order)


@pytest.mark.parametrize(
    "column",
    [
        "distance_km",
        "items",
        "hour",
        "courier_months",
        "weather",
        "day_type",
        "vehicle",
    ],
)
def test_missing_required_field_is_rejected(valid_order, column):
    valid_order[column] = None
    with pytest.raises(ValueError, match=column):
        predict.predict_order(valid_order)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("weather", "Snow"),
        ("weather", "hail"),
        ("day_type", "holiday"),
        ("vehicle", "truck"),
    ],
)
def test_unseen_category_is_rejected(valid_order, column, bad_value):
    valid_order[column] = bad_value
    with pytest.raises(ValueError, match=column):
        predict.predict_order(valid_order)


def test_column_the_model_does_not_use_is_rejected(valid_order):
    valid_order["customer_rating"] = 5
    with pytest.raises(ValueError, match="customer_rating"):
        predict.predict_order(valid_order)


def test_choices_match_what_the_model_was_trained_on():
    encoder = predict.PIPELINE.named_steps["prep"].named_transformers_["cat"]
    trained = dict(zip(predict.CATEGORICAL, encoder.categories_, strict=True))
    for column, allowed in predict.CHOICES.items():
        assert sorted(allowed) == sorted(trained[column])


def test_model_file_is_loaded_once(monkeypatch, valid_order):
    calls = []
    real_load = joblib.load

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)
    module = importlib.reload(predict)  # importing loads the model one time
    for _ in range(3):
        module.predict_order(valid_order)
    assert len(calls) == 1
