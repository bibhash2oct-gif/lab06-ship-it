"""QuickBite late order check: a Gradio page that calls predict.py only."""

import gradio as gr

import predict

# The free ZeroGPU hardware refuses to start without one GPU function. This model
# runs on the CPU, so the function does nothing. On your own computer the spaces
# package is not installed, and the ImportError is ignored.
try:
    import spaces

    @spaces.GPU
    def gpu_placeholder():
        return None

except ImportError:
    pass

LABELS = {
    "distance_km": "Distance (km)",
    "items": "Number of items",
    "hour": "Hour the order is placed (24-hour clock)",
    "prep_minutes": "Kitchen prep time (minutes, may be left empty)",
    "courier_months": "Courier experience (months)",
    "weather": "Weather",
    "day_type": "Day type",
    "vehicle": "Courier vehicle",
}


def make_field(name: str):
    """One input field per column in model_info.json."""
    if name in predict.CHOICES:
        return gr.Dropdown(
            choices=predict.CHOICES[name], value=None, label=LABELS[name]
        )
    low, high, whole = predict.NUMBER_RULES[name]
    kind = "a whole number" if whole else "a number"
    return gr.Number(
        value=None, label=LABELS[name], info=f"Enter {kind} from {low} to {high}."
    )


def check_and_score(*values):
    order = dict(zip(predict.FEATURES, values, strict=True))
    try:
        result = predict.predict_order(order)
    except ValueError as error:
        raise gr.Error(f"Please fix this field: {error}") from error
    for message in result["warnings"]:
        gr.Warning(message)

    chance = result["probability"]
    line = result["threshold"]
    chances = {"Late": chance, "On time": 1 - chance}
    if result["offer_voucher"]:
        action = (
            f"**Offer the $4 voucher now.** The chance this order arrives late is "
            f"{chance:.0%}, at or above the {line:.0%} line where a voucher pays off."
        )
    else:
        action = (
            f"**No voucher needed.** The chance this order arrives late is "
            f"{chance:.0%}, below the {line:.0%} line where a voucher pays off."
        )
    return chances, action


with gr.Blocks(title="QuickBite Late Order Check") as demo:
    gr.Markdown(
        "# QuickBite late order check\n"
        "Enter what you know about a new order. The page says how likely it is to "
        "arrive late, and whether to offer the customer a $4 voucher now."
    )
    fields = [make_field(name) for name in predict.FEATURES]
    button = gr.Button("Check this order", variant="primary")
    chances = gr.Label(label="Chances", num_top_classes=2)
    action = gr.Markdown()
    button.click(check_and_score, inputs=fields, outputs=[chances, action])

if __name__ == "__main__":
    demo.launch()
