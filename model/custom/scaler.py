class Scaler:
    def __init__(self, kind="simple", feature_second=False):
        self.kind = kind
        self.feature_second = feature_second

    def fit_transform(self, x):
        if self.feature_second:
            scale = x.mean(dim=2, keepdim=True)
        else:
            scale = x.mean(dim=1, keepdim=True)

        x = x / scale

        return x, scale

    def transform(self, x, scale):
        x = x / scale

        return x

    def invert_transform(self, x, scale):
        x = x * scale

        return x