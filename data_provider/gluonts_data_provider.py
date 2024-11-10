from gluonts.dataset.common import ListDataset

from data_provider.data_loader import DataLoader

class GluontsDataProvider(DataLoader):
    def train_dataset(self):
        return self.gluonts_data_formatter(self.train)

    def test_dataset(self):
        return self.gluonts_data_formatter(self.test)

    def test_length(self):
        return len(self.test) - self.context_length - self.prediction_length

    def test_prediction_data(self, num_segment: int):
        target_data = self.gluonts_data_formatter(self.test.iloc[num_segment:num_segment + self.context_length])
        correct_data = self.gluonts_data_formatter(self.test.iloc[num_segment + self.context_length:num_segment + self.context_length + self.prediction_length])

        return target_data, correct_data
        
    def gluonts_data_formatter(self, dataset):
        return ListDataset(
            [{"start": dataset.index[0], "target": dataset["close"]}],
            freq=self.freq,
        )

    def listdata_values(self, data):
        return data[0]["target"]