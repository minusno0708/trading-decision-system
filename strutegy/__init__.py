import random

def random() -> str:
    return random.choice(["buy", "sell", "hold"])

def diff_next_mean(history_values: list, forecast_values: list, threshold_value = 0) -> str:
    current_price = history_values[-1]
    next_day_price = forecast_values[0]

    price_diff = next_day_price - current_price

    if price_diff > threshold_value:
        return "buy"
    elif price_diff < threshold_value * -1:
        return "sell"
    else:
        return "hold"

def diff_all_mean(history_values: list, forecast_values: list, threshold_value = 0) -> str:
    history_mean = sum(history_values) / len(history_values)
    forecast_mean = sum(forecast_values) / len(forecast_values)

    price_diff = forecast_mean - history_mean

    if price_diff > threshold_value:
        return "buy"
    elif price_diff < threshold_value * -1:
        return "sell"
    else:
        return "hold"