import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model.custom.estimator import Estimator
from model.custom.output import ForecastOutput

class Model:
    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            freq: str = "D",
            epochs: int = 100,
            num_parallel_samples: int = 1000,
            target_dim: int = 1,
            is_scaling: bool = False,
            add_time_features: bool = True,
            add_extention_features: bool = True
        ):
        self.is_scaling = is_scaling
        self.feature_second = False

        self.add_time_features = add_time_features
        self.num_time_features = 4

        self.add_extention_features = add_extention_features
        self.num_extention_features = 3

        if self.feature_second:
            self.input_length = context_length
            self.output_length = prediction_length
        else:
            self.input_length = target_dim
            self.output_length = target_dim

            if self.add_time_features:
                self.input_length += self.num_time_features
            if self.add_extention_features:
                self.input_length += self.num_extention_features

        self.freq = freq
        self.epochs = epochs
        self.num_parallel_samples = num_parallel_samples
        
        self.model = Estimator(
            input_size=self.input_length,
            output_size=self.output_length,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.GaussianNLLLoss()

    def permute_dim(self, x):
        # 元の次元 [batch_size, feature_size, time_step]
        if self.feature_second:
            return x.permute(0, 1, 2)
        else:
            # 変換後 [batch_size, time_step, feature_size]
            return x.permute(0, 2, 1)

    def scaling(self, x):
        if self.feature_second:
            scale = x.mean(dim=2, keepdim=True)
        else:
            scale = x.mean(dim=1, keepdim=True)
        x = x / scale

        return x, scale

    def rescaling(self, mean, var, scale):
        mean = mean * scale
        var = var * scale ** 2

        return mean, var

    def train(self, dataset: torch.utils.data.DataLoader, val_dataset: torch.utils.data.DataLoader = None):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        train_loss = []
        val_loss = []

        for epoch in range(self.epochs):
            self.model.train()

            train_loss_per_epoch = np.array([])
            
            minimal_val_loss = {"loss": np.inf, "epoch": 0}

            for i, (start_date, input_x, target_x, time_features, extention_features) in enumerate(dataset):
                optimizer.zero_grad()
                batch_size = input_x.shape[0]
                hidden = self.model.init_hidden(batch_size)

                input_x = self.permute_dim(input_x).to(self.device)
                target_x = self.permute_dim(target_x).to(self.device)
                time_features = time_features.to(self.device)
                extention_features = extention_features.to(self.device)

                if self.is_scaling:
                    input_x, scale = self.scaling(input_x)

                if self.add_time_features:
                    input_x = torch.cat([input_x, time_features], dim=2)

                if self.add_extention_features:
                    input_x = torch.cat([input_x, extention_features], dim=2)

                mean, var = self.model(input_x, hidden)

                if self.is_scaling:
                    mean, var = self.rescaling(mean, var, scale)
                
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
                    for i, (start_date, input_x, target_x, time_features, extention_features) in enumerate(val_dataset):
                        batch_size = input_x.shape[0]
                        hidden = self.model.init_hidden(batch_size)

                        input_x = self.permute_dim(input_x).to(self.device)
                        target_x = self.permute_dim(target_x).to(self.device)
                        time_features = time_features.to(self.device)
                        extention_features = extention_features.to(self.device)

                        if self.is_scaling:
                            input_x, scale = self.scaling(input_x)

                        if self.add_time_features:
                            input_x = torch.cat([input_x, time_features], dim=2)

                        if self.add_extention_features:
                            input_x = torch.cat([input_x, extention_features], dim=2)

                        mean, var = self.model(input_x, hidden)

                        if self.is_scaling:
                            mean, var = self.rescaling(mean, var, scale)
                        
                        loss = self.criterion(mean, target_x, var)
                        val_loss_per_epoch = np.append(val_loss_per_epoch, loss.item())

                        if loss.item() < minimal_val_loss["loss"]:
                            minimal_val_loss["loss"] = loss.item()
                            minimal_val_loss["epoch"] = epoch
            
                loss_mean = np.mean(val_loss_per_epoch)
                val_loss.append(loss_mean)

                print(f'Epoch [{epoch+1}/{self.epochs}], train_Loss: {train_loss[-1]:.4f}, val_loss: {val_loss[-1]:.4f}')
            else:
                print(f'Epoch [{epoch+1}/{self.epochs}], train_Loss: {train_loss[-1]:.4f}')

        return train_loss, val_loss, minimal_val_loss

    def forecast(self, input_x: torch.tensor, time_features: torch.tensor, extention_features: torch.tensor):
        self.model.eval()

        input_x = self.permute_dim(input_x).to(self.device)
        time_features = time_features.to(self.device)
        extention_features = extention_features.to(self.device)

        with torch.no_grad():
            if self.is_scaling:
                input_x, scale = self.scaling(input_x)

            if self.add_time_features:
                input_x = torch.cat([input_x, time_features], dim=2)

            if self.add_extention_features:
                input_x = torch.cat([input_x, extention_features], dim=2)

            mean, var = self.model(input_x)

            if self.is_scaling:
                mean, var = self.rescaling(mean, var, scale)

        mean = mean.cpu().numpy()
        var = var.cpu().numpy()

        output = ForecastOutput(mean, var, self.num_parallel_samples)
        
        return output

    def make_evaluation_predictions(self, input_x: torch.tensor, target_x: torch.tensor, time_features: torch.tensor, extention_features: torch.tensor):
        self.model.eval()

        input_x = self.permute_dim(input_x).to(self.device)
        target_x = self.permute_dim(target_x).to(self.device)

        time_features = time_features.to(self.device)
        extention_features = extention_features.to(self.device)

        with torch.no_grad():
            if self.is_scaling:
                input_x, scale = self.scaling(input_x)

            if self.add_time_features:
                input_x = torch.cat([input_x, time_features], dim=2)

            if self.add_extention_features:
                input_x = torch.cat([input_x, extention_features], dim=2)

            mean, var = self.model(input_x)

            if self.is_scaling:
                mean, var = self.rescaling(mean, var, scale)

            loss = self.criterion(mean, target_x, var)

        mean = self.permute_dim(mean).squeeze(0).cpu().numpy()
        var = self.permute_dim(var).squeeze(0).cpu().numpy()
        loss = loss.cpu().numpy()

        output = []

        for i in range(len(mean)):
            output.append(ForecastOutput(mean[i], var[i], self.num_parallel_samples))

        return output, loss
    
        


