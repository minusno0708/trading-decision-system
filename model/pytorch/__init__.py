import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model.pytorch.estimator import Estimator

class Model:
    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            freq: str = "D",
            epochs: int = 100,
            num_parallel_samples: int = 1000,
            model_name="deepar",
            model_type="torch"
        ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        self.epochs = epochs
        self.num_parallel_samples = num_parallel_samples
        
        self.model = Estimator(
            input_size=self.context_length,
            output_size=self.prediction_length,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        

    def train(self, dataset: torch.utils.data.DataLoader):
        criterion = nn.GaussianNLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()

        for epoch in range(self.epochs):
            for i, (start_date, input_x, target_x, time_feature) in enumerate(dataset):
                optimizer.zero_grad()
                #batch_size = train.shape[0]
                #hidden = self.model.init_hidden(batch_size)  # Initialize hidden state

                input_x = input_x.to(self.device)
                target_x = target_x.to(self.device)
                time_feature = time_feature.to(self.device)

                mean, var = self.model(input_x)
                
                loss = criterion(mean, target_x, var)
                
                # モデルの更新
                loss.backward()
                optimizer.step()
                
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

    def forecast(self, input_x: torch.tensor):
        self.model.eval()

        input_x = input_x.to(self.device)

        with torch.no_grad():
            mean, var = self.model(input_x)

        mean = mean.cpu().numpy()
        var = var.cpu().numpy()
        
        return mean, var
        


