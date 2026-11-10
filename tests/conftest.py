"""Shared fixtures: a valid order, and one clearly easy and one clearly hard order."""

import pytest


@pytest.fixture
def valid_order() -> dict:
    return {
        "distance_km": 4.2,
        "items": 3,
        "hour": 19,
        "prep_minutes": 15,
        "courier_months": 12,
        "weather": "rain",
        "day_type": "weekend",
        "vehicle": "bike",
    }


@pytest.fixture
def easy_order() -> dict:
    """Short trip, small order, quiet hour, clear weather, experienced courier."""
    return {
        "distance_km": 1.0,
        "items": 1,
        "hour": 15,
        "prep_minutes": 8,
        "courier_months": 59,
        "weather": "clear",
        "day_type": "weekday",
        "vehicle": "car",
    }


@pytest.fixture
def hard_order() -> dict:
    """Long trip, big order, dinner rush, snow, weekend, a new courier on a bike."""
    return {
        "distance_km": 14.0,
        "items": 8,
        "hour": 19,
        "prep_minutes": 35,
        "courier_months": 0,
        "weather": "snow",
        "day_type": "weekend",
        "vehicle": "bike",
    }
