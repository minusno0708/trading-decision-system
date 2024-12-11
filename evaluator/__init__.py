import numpy as np

class Evaluator:
    def __init__(self, target = ["rmse", "quantile_loss", "coverage"], quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        self.target = target
        self.quantiles = quantiles

        self.metrics = {}
        for t in target:
            if t in ["quantile_loss", "w_quantile_loss", "coverage"]:
                for q in quantiles:
                    self.metrics[t + f"[{str(q)}]"] = 0
            else:
                self.metrics[t] = 0
            self.metrics["count"] = 0

    def evaluate(self, forecasts, target):
        metrics = {}
        
        if "abs_target_sum" in self.target:
            metrics['abs_target_sum'] = self.abs_target_sum(target)
        if "abs_target_mean" in self.target:
            metrics['abs_target_mean'] = self.abs_target_mean(target)
        if "mae" in self.target:
            metrics['mae'] = self.mae(target, forecasts.mean)
        if "rmse" in self.target:
            metrics['rmse'] = self.rmse(target, forecasts.mean)

        for q in self.quantiles:
            if "quantile_loss" in self.target:
                metrics[f'quantile_loss[{str(q)}]'] = self.quantile_loss(target, forecasts.quantile(q), q)
            if "w_quantile_loss" in self.target:
                metrics[f'w_quantile_loss[{str(q)}]'] = self.w_quantile_loss(metrics[f'quantile_loss[{str(q)}]'], metrics['abs_target_sum'])
            if "coverage" in self.target:
                metrics[f'coverage[{str(q)}]'] = self.coverage(target, forecasts.quantile(q))

        self.update(metrics)

        return metrics

    def update(self, metrics):
        for t in self.target:
            if t in ["quantile_loss", "w_quantile_loss", "coverage"]:
                for q in self.quantiles:
                    self.metrics[t + f"[{str(q)}]"] += metrics[t + f"[{str(q)}]"]
            else:
                self.metrics[t] += metrics[t]
        self.metrics["count"] += 1

    def get_metrics(self):
        metrics = {}
        for t in self.target:
            if t in ["quantile_loss", "w_quantile_loss", "coverage"]:
                for q in self.quantiles:
                    metrics[t + f"[{str(q)}]"] = self.metrics[t + f"[{str(q)}]"] / self.metrics["count"]
            else:
                metrics[t] = self.metrics[t] / self.metrics["count"]
        return metrics

    def abs_target_sum(self, target):
        return np.sum(np.abs(target))

    def abs_target_mean(self, target):
        return np.mean(np.abs(target))

    def mae(self, target, forecast):
        return np.mean(np.abs(target - forecast))

    def rmse(self, target, forecast):
        return np.sqrt(np.mean(np.square((target - forecast))))

    def nrmse(self, rmse, abs_target_mean):
        return rmse / abs_target_mean

    def quantile_loss(self, target, forecast, q):
        return 2 * np.sum(np.abs((forecast - target) * (target <= forecast) - q))

    def w_quantile_loss(self, quantile_loss, abs_target_sum):
        return quantile_loss / abs_target_sum

    def coverage(self, target, forecast):
        return np.mean(target <= forecast)