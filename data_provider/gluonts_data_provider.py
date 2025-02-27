from gluonts.dataset.common import ListDataset

from data_provider.data_loader import DataLoader

class GluontsDataProvider(DataLoader):
    def train_dataset(self):
        return self.gluonts_data_formatter(self.train)

    def test_dataset(self):
        return self.gluonts_data_formatter(self.test)

    def data_length(self, target: str = "test"):
        if target == "train":
            return len(self.train) - self.context_length - self.prediction_length
        else:
            return len(self.test) - self.context_length - self.prediction_length

    def evaluation_data(self, num_segment: int, target: str = "test"):
        if target == "train":
            return self.gluonts_data_formatter(self.train.iloc[num_segment:num_segment + self.context_length + self.prediction_length])
        else:
            return self.gluonts_data_formatter(self.test.iloc[num_segment:num_segment + self.context_length + self.prediction_length])

    def prediction_data(self, num_segment: int, target: str = "test"):
        if target == "train":
            target_data = self.gluonts_data_formatter(self.train.iloc[num_segment:num_segment + self.context_length])
            correct_data = self.gluonts_data_formatter(self.train.iloc[num_segment + self.context_length:num_segment + self.context_length + self.prediction_length])
        else:
            target_data = self.gluonts_data_formatter(self.test.iloc[num_segment:num_segment + self.context_length])
            correct_data = self.gluonts_data_formatter(self.test.iloc[num_segment + self.context_length:num_segment + self.context_length + self.prediction_length])

        return target_data, correct_data
        
    def gluonts_data_formatter(self, dataset):
        return ListDataset(
            [{"start": dataset.index[0], "target": dataset[col]} for col in self.target_cols],
            freq=self.freq,
        )

    def listdata_values(self, data):
        return data[0]["target"]

    def listdata_dates(self, data):
        start_date = data[0]["start"]
        date_list = []

        for i in range(len(data[0]["target"])):
            date_list.append((start_date + i).to_timestamp())

        return date_list