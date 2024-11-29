import numpy as np

class ForecastOutput:
    def __init__(self, mean, var, num_samples = 100):
        self.distribution = {
            "mean": mean,
            "var": var
        }
        self.num_samples = num_samples
        #self.samples = self.gen_samples()

    def gen_samples(self):
        samples = np.random.normal(self.distribution["mean"], self.distribution["var"], self.num_samples)
        return samples