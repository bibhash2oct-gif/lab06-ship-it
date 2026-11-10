---
title: QuickBite Late Order Check
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
pinned: false
short_description: Will this order arrive late?
---

# QuickBite Late Order Check

## What it does
A dispatcher enters what is known about a new order, before a courier is assigned.
The page shows the chance that the order arrives late and says whether to offer the
customer a $4 voucher now. It uses the model and threshold from Lab 05: a voucher is
offered when the chance of a late delivery is 30% or more.

Live page: https://huggingface.co/spaces/bibhashdas/lab06-ship-it

## How to run it locally
~~~
conda env create -f environment.yml
conda activate lab06-ship-it
gradio app.py
~~~
Then open http://127.0.0.1:7860 in a browser.

## Decisions
Every input is checked in `predict.py` before it reaches the model. The limits follow
the range the model was trained on: far outside it, the model has no basis for an
answer, so the page refuses and says which field is wrong and why.

| Field | Allowed | Why |
|---|---|---|
| distance_km | 0.1 to 30 | Training orders ran from 0.3 to 15 km. Zero or more than 30 km is a typing mistake or a trip the model has never seen. |
| items | Whole number, 1 to 20 | An order has at least one item, and half an item is impossible. Training orders had 1 to 8 items. |
| hour | Whole number, 10 to 22 | QuickBite takes orders from 10:00 to 22:59, the hours in the training data. |
| prep_minutes | 1 to 60, or empty | Training values ran from 4 to 40. It is often not known yet, so empty is accepted: the pipeline fills in the training median and the page shows a warning. |
| courier_months | 0 to 120 | 0 is a brand-new courier. Negative experience is impossible. Training values ran from 0 to 59. |
| weather | clear, rain, snow | The only values the model was trained on. Anything else is refused (see below). |
| day_type | weekday, weekend | The only values the model was trained on. |
| vehicle | bike, scooter, car | The only values the model was trained on. |
| Any other column | Refused | The page asks only for what is known when the order is placed. customer_rating, for example, exists only after delivery. |

**Why unseen choices are refused.** The encoder in the pipeline ignores values it has
not seen, so a typo does not crash: it scores the order as if there were no weather at
all. Example: 6 km, 3 items, 15:00, 14 minutes prep, a courier with 24 months, on a
weekday, by scooter. With weather `snow` the chance of a late delivery is 31.6% and
the voucher is offered. With `Snow` it is 26.1% and the voucher is not offered.
The page would look fine and the answer would be wrong. `predict.py` refuses `Snow`,
and the page only offers the three trained choices.

## Result
From the Lab 05 test set: 750 orders the model never saw, scored once at
threshold 0.3.

|  | Predicted on time | Predicted late |
|---|---|---|
| Actually on time | 578 | 55 |
| Actually late | 64 | 53 |

Precision 0.491, recall 0.453, F1 0.471, ROC-AUC 0.796.

| Policy | Cost on the test set |
|---|---|
| Flag nobody (today) | $1,755 |
| Flag everybody | $3,000 |
| Use the model | $1,392 |

The model saves $363 (21%) against flagging nobody.

## Tests
- `tests/test_predict.py`: the probability is between 0 and 1; a clearly harder order
  scores higher than a clearly easier one; the voucher decision exactly at, just below
  and just above the threshold (worked out by hand); the threshold comes from
  model_info.json; an empty prep_minutes is accepted with a warning; every impossible
  number, missing field, unseen choice and extra column is refused with a message
  naming the field; the choices match what the model was trained on; the model file
  is loaded once (joblib.load counted with monkeypatch).
- `tests/test_deployment.py`: the installed scikit-learn equals the version in
  model_info.json and the pin in requirements.txt; installed versions match the pins;
  gradio is not in requirements.txt; sdk_version equals the installed Gradio; the
  app_file exists; requirements-dev.txt pins gradio, pytest and ruff; models/ is not
  ignored by git.

~~~
python -m ruff format --check .
python -m ruff check .
python -m pytest
~~~

## Limitations
- The model learned from 3,000 invented orders made for the course, not from real
  QuickBite deliveries. Its numbers show the method, not real-world accuracy.
- It cannot see anything outside the eight fields: traffic, a road closure, an
  accident, a restaurant running behind, or a big event nearby.
- prep_minutes in training was the real kitchen time. When an order is placed only an
  estimate exists, so answers using an estimate may be less accurate.
- It is often wrong about single orders: about half the flagged orders arrive on time,
  and about half the late orders are not flagged. It lowers the total cost of
  vouchers and late orders; it does not promise anything about one order.
- Do not use it to judge or rank couriers, to promise delivery times to customers, or
  for orders outside the limits above.
