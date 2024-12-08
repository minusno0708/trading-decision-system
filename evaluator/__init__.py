import numpy as np

class Evaluator:
    def __init__(self, quantiles = [0.1, 0.5, 0.9]):
        self.quantiles = quantiles

    def evaluate(self, forecasts, target):
        metrics = {}

        metrics['abs_target_sum'] = self.abs_target_sum(target)
        metrics['abs_target_mean'] = self.abs_target_mean(target)

        metrics['rmse'] = self.rmse(target, forecasts.values.mean)

        metrics['nrmse'] = self.nrmse(metrics['rmse'], metrics['abs_target_mean'])

        for q in self.quantiles:
            metrics[f'quantile_loss[{str(q)}]'] = self.quantile_loss(target, forecasts.values.quantile(q), q)
            metrics[f'w_quantile_loss[{str(q)}]'] = self.w_quantile_loss(metrics[f'quantile_loss[{str(q)}]'], metrics['abs_target_sum'])
            metrics[f'coverage[{str(q)}]'] = self.coverage(target, forecasts.values.quantile(q))

        return metrics

    def abs_target_sum(self, target):
        return np.sum(np.abs(target))

    def abs_target_mean(self, target):
        return np.mean(np.abs(target))

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