"""Load the Lab 05 pipeline once, check one order, score it and decide.

Everything the page shows goes through this module. Only this module talks to the
model.
"""

import json
import math
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Loaded once, when the module is imported, never on each prediction.
MODEL_INFO = json.loads((MODELS_DIR / "model_info.json").read_text(encoding="utf-8"))
PIPELINE = joblib.load(MODELS_DIR / "late_orders.joblib")

# The columns and the threshold come from model_info.json, not from typed lists.
NUMERIC = list(MODEL_INFO["numeric_features"])
CATEGORICAL = list(MODEL_INFO["categorical_features"])
FEATURES = NUMERIC + CATEGORICAL
THRESHOLD = float(MODEL_INFO["threshold"])

# The limits are decisions, explained in the README: (lowest, highest, whole number).
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
# The only field allowed to be empty. The pipeline fills it with the training median.
OPTIONAL = {"prep_minutes"}

if set(FEATURES) != set(NUMBER_RULES) | set(CHOICES):
    raise RuntimeError("model_info.json lists columns that predict.py has no rule for")


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return isinstance(value, float) and math.isnan(value)


def check_order(order: dict) -> tuple[dict, list[str]]:
    """Return a clean copy of the order and any warnings.

    Raises ValueError naming the field and the rule when anything is wrong.
    """
    unknown = sorted(set(order) - set(FEATURES))
    if unknown:
        raise ValueError(f"{unknown[0]} is not an input the model uses")

    clean: dict = {}
    warnings: list[str] = []
    for name in FEATURES:
        value = order.get(name)
        if _is_empty(value):
            if name in OPTIONAL:
                clean[name] = float("nan")
                warnings.append(
                    f"{name} is empty, so the model uses the typical value from "
                    "its training data. The answer is less certain."
                )
                continue
            raise ValueError(
                f"{name} is missing: every field except prep_minutes is required"
            )

        if name in CHOICES:
            allowed = CHOICES[name]
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(
                    f"{name} must be one of {', '.join(allowed)} (got {value!r})"
                )
            clean[name] = value
            continue

        low, high, whole = NUMBER_RULES[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number (got {value!r})")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a real number (got {value!r})")
        if whole and not float(value).is_integer():
            raise ValueError(f"{name} must be a whole number (got {value!r})")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high} (got {value!r})")
        clean[name] = float(value)
    return clean, warnings


def decide(probability: float, threshold: float = THRESHOLD) -> bool:
    """Offer the voucher when the chance of a late delivery is at or above the line."""
    return probability >= threshold


def predict_order(order: dict) -> dict:
    """Check one order, score it and return the probability, decision and threshold."""
    clean, warnings = check_order(order)
    row = pd.DataFrame([clean], columns=FEATURES)
    probability = float(PIPELINE.predict_proba(row)[0, 1])
    return {
        "probability": probability,
        "offer_voucher": decide(probability),
        "threshold": THRESHOLD,
        "warnings": warnings,
    }
