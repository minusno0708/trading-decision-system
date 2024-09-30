import random

def random_decision() -> str:
    return random.choice(["buy", "sell"])

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
    current_price = history_values[-1]
    forecast_mean = sum(forecast_values) / len(forecast_values)

    price_diff = forecast_mean - current_price

    if price_diff > threshold_value:
        return "buy"
    elif price_diff < threshold_value * -1:
        return "sell"
    else:
        return "hold"

def all_win(history_values: list, next_price: float) -> str:
    current_price = history_values[-1]

    if current_price < next_price:
        return "buy"
    else:
        return "sell"

def all_lose(history_values: list, next_price: float) -> str:
    current_price = history_values[-1]

    if current_price > next_price:
        return "buy"
    else:
        return "sell"