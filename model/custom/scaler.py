import torch

class Scaler:
    def __init__(self, name="mean", feature_second=False):
        self.name = name
        self.feature_second = feature_second

        if self.feature_second:
            self.target_dim = 2
        else:
            self.target_dim = 1

    def fit(self, x):
        if self.name == "mean":
            scale = x.mean(dim=self.target_dim, keepdim=True)
        elif self.name == "abs_mean":
            scale = x.mean(dim=self.target_dim, keepdim=True).abs()
            scale = scale + 1
        elif self.name == "const":
            scale_num = 1

            batch_size = x.size(0)
            device = x.device

            scale = torch.ones(batch_size, 1, 1) * scale_num
            scale = scale.to(device)
        elif self.name == "standard":
            mean = x.mean(dim=self.target_dim, keepdim=True)
            std = x.std(dim=self.target_dim, keepdim=True)

            scale = (mean, std)
        else:
            raise ValueError("Invalid scaler name")

        return scale

    def transform(self, x, scale):
        if self.name == "standard":
            x = (x - scale[0]) / scale[1]
        elif self.name == "mean":
            x = x - scale
        else:
            x = x / scale

        return x

    def fit_transform(self, x):
        scale = self.fit(x)

        x = self.transform(x, scale)

        return x, scale

    def invert_transform(self, mean, var, scale):
        if self.name == "standard":
            mean = mean * scale[1] + scale[0]
            var = var * (scale[1] ** 2)
        elif self.name == "mean":
            mean = mean + scale
        else:
            mean = mean * scale
            var = var * (scale ** 2)

        return mean, var