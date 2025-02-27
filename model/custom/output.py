import numpy as np

class ForecastOutput:
    def __init__(self, mean, var, num_samples = 100):
        self.distribution = {
            "mean": mean,
            "var": var
        }
        self.length = mean.shape[0]
        self.num_samples = num_samples
        self.samples = self.gen_samples(mean, var)
        self.mean = self.gen_mean()
        self.median = self.quantile(0.5)

    def gen_samples(self, mean, var):
        samples = np.array([])
        for i in range(self.length):
            samples = np.append(samples, np.random.normal(mean[i], var[i], self.num_samples))

        samples = samples.reshape(self.length, self.num_samples)

        return samples

    def gen_mean(self):
        mean = np.array([])
        for i in range(self.length):
            mean = np.append(mean, np.mean(self.samples[i]))
        return mean

    def quantile(self, q):
        return np.quantile(self.samples, q, axis=1)

    def inverse_transform(self, scaler):
        self.distribution["mean"] = scaler.inverse_transform([self.distribution["mean"]]).reshape(-1)
        self.distribution["var"] = scaler.inverse_transform([self.distribution["var"]]).reshape(-1)
        self.samples = scaler.inverse_transform(self.samples)
        self.mean = self.gen_mean()
        self.median = self.quantile(0.5)