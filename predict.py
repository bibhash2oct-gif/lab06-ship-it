"""Check order inputs and predict late delivery using the saved Lab 05 model."""

import json
import math
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_INFO = json.loads(
    (ROOT / "models" / "model_info.json").read_text(encoding="utf-8")
)
NUMERIC = MODEL_INFO["numeric_features"]
CATEGORICAL = MODEL_INFO["categorical_features"]
FEATURES = NUMERIC + CATEGORICAL
THRESHOLD = MODEL_INFO["threshold"]
PIPELINE = joblib.load(ROOT / "models" / "late_orders.joblib")

NUMBER_RULES = {
    "distance_km": (0.1, 30, False),
    "items": (1, 20, True),
    "hour": (10, 22, True),
    "prep_minutes": (1, 60, False),
    "courier_months": (0, 120, False),
}
CHOICES = {
    "weather": ["clear", "rain", "snow"],
    "day_type": ["weekday", "weekend"],
    "vehicle": ["bike", "scooter", "car"],
}


def check_inputs(order: dict) -> dict:
    """Return checked inputs, or raise ValueError naming the field and rule."""
    for name in order:
        if name not in FEATURES:
            raise ValueError(f"{name} is not an input used by this model.")

    checked = {}
    for name in NUMERIC:
        value = order.get(name)
        empty = (
            value is None
            or (isinstance(value, str) and not value.strip())
            or (isinstance(value, Real) and value != value)
        )
        if name == "prep_minutes" and empty:
            checked[name] = float("nan")  # The pipeline fills in the training median.
            continue

        low, high, whole = NUMBER_RULES[name]
        kind = "a whole number" if whole else "a number"
        rule = f"{name} must be {kind} from {low} to {high}"
        rule += (
            ", or left empty." if name == "prep_minutes" else "; it cannot be empty."
        )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(rule)
        if not low <= value <= high:
            raise ValueError(rule)
        if whole and not float(value).is_integer():
            raise ValueError(rule)
        checked[name] = int(value) if whole else float(value)

    for name in CATEGORICAL:
        value = order.get(name)
        if not isinstance(value, str) or value not in CHOICES[name]:
            raise ValueError(f"{name} must be one of: {', '.join(CHOICES[name])}.")
        checked[name] = value
    return checked


def decide(probability: float, threshold: float | None = None) -> bool:
    """Offer a voucher when the probability meets or exceeds the threshold."""
    if threshold is None:
        threshold = THRESHOLD
    return bool(probability >= threshold)


def predict_order(order: dict) -> dict:
    """Check and score one order, returning the decision and any warning."""
    checked = check_inputs(order)
    row = pd.DataFrame([checked], columns=FEATURES)
    probability = float(PIPELINE.predict_proba(row)[0, 1])
    warnings = []
    if math.isnan(checked["prep_minutes"]):
        warnings.append(
            "prep_minutes was left empty; the pipeline used the training median."
        )
    return {
        "probability": probability,
        "offer_voucher": decide(probability),
        "threshold": THRESHOLD,
        "warnings": warnings,
    }
