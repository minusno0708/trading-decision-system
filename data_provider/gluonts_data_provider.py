from gluonts.dataset.common import ListDataset

from data_provider.data_loader import DataLoader

class GluontsDataProvider(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.train = self.gluonts_data_formatter(self.train)
        self.test = self.gluonts_data_formatter(self.test)

    def gluonts_data_formatter(self, dataset):
        return ListDataset(
            [{"start": dataset.index[0], "target": dataset["close"]}],
            freq=self.freq,
        )