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

def all_true(history_values: list, next_price: float) -> str:
    current_price = history_values[-1]

    if current_price < next_price:
        return "buy"
    else:
        return "sell"

def all_false(history_values: list, next_price: float) -> str:
    current_price = history_values[-1]

    if current_price > next_price:
        return "buy"
    else:
        return "sell"

class Strategy:
    def __init__(self, decision_method: str):
        self.decision_method = decision_method

    def decide_action(self, history_values: list, forecast_values: list, future_values: list = None) -> str:
        if self.decision_method == "diff_next_mean":
            action = diff_next_mean(history_values, forecast_values)
        elif self.decision_method == "random":
            action = random_decision()
        elif self.decision_method == "all_win":
            action = all_true(history_values, future_values[0])
        elif self.decision_method == "all_lose":
            action = all_false(history_values, future_values[0])
        elif self.decision_method == "all_buy":
            action = "buy"
        elif self.decision_method == "all_sell":
            action = "sell"
        elif self.decision_method == "cross_action":
            if not "action" in locals():
                action = "sell"
            elif action == "buy":
                action = "sell"
            elif action == "sell":
                action = "buy"

        return action