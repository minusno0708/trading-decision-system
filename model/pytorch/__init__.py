import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model.pytorch.estimator import Estimator
from model.pytorch.output import ForecastOutput

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

        self.criterion = nn.GaussianNLLLoss()

    def train(self, dataset: torch.utils.data.DataLoader, val_dataset: torch.utils.data.DataLoader = None):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        train_loss = []
        val_loss = []

        for epoch in range(self.epochs):
            self.model.train()

            train_loss_per_epoch = np.array([])
            for i, (start_date, input_x, target_x, time_feature) in enumerate(dataset):
                optimizer.zero_grad()
                batch_size = input_x.shape[0]
                hidden = self.model.init_hidden(batch_size)

                input_x = input_x.to(self.device)
                target_x = target_x.to(self.device)
                time_feature = time_feature.to(self.device)

                mean, var = self.model(input_x, hidden)
                
                loss = self.criterion(mean, target_x, var)
                train_loss_per_epoch = np.append(train_loss_per_epoch, loss.item())
                
                # モデルの更新
                loss.backward()
                optimizer.step()
            
            loss_mean = np.mean(train_loss_per_epoch)
            train_loss.append(loss_mean)

            # 検証データのロスを計算
            if val_dataset is not None:
                self.model.eval()

                val_loss_per_epoch = np.array([])
                with torch.no_grad():
                    for i, (start_date, input_x, target_x, time_feature) in enumerate(val_dataset):
                        batch_size = input_x.shape[0]
                        hidden = self.model.init_hidden(batch_size)

                        input_x = input_x.to(self.device)
                        target_x = target_x.to(self.device)
                        time_feature = time_feature.to(self.device)

                        mean, var = self.model(input_x, hidden)
                        
                        loss = self.criterion(mean, target_x, var)
                        val_loss_per_epoch = np.append(val_loss_per_epoch, loss.item())
            
                loss_mean = np.mean(val_loss_per_epoch)
                val_loss.append(loss_mean)

                print(f'Epoch [{epoch+1}/{self.epochs}], train_Loss: {train_loss[-1]:.4f}, val_loss: {val_loss[-1]:.4f}')
            else:
                print(f'Epoch [{epoch+1}/{self.epochs}], train_Loss: {train_loss[-1]:.4f}')

        return train_loss, val_loss

    def forecast(self, input_x: torch.tensor):
        self.model.eval()

        input_x = input_x.to(self.device)

        with torch.no_grad():
            mean, var = self.model(input_x)

        mean = mean.cpu().numpy()
        var = var.cpu().numpy()

        output = ForecastOutput(mean, var, self.num_parallel_samples)
        
        return output

    def make_evaluation_predictions(self, input: torch.tensor, target: torch.tensor):
        self.model.eval()

        input = input.to(self.device)
        target = target.to(self.device)

        with torch.no_grad():
            mean, var = self.model(input)
            loss = self.criterion(mean, target, var)

        mean = mean.cpu().numpy()
        var = var.cpu().numpy()
        loss = loss.cpu().numpy()

        output = ForecastOutput(mean, var, self.num_parallel_samples)

        return output, loss
        


