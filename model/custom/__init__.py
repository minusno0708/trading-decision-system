import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model.custom.estimator import Estimator
from model.custom.output import ForecastOutput
from model.custom.scaler import Scaler
from model.custom.loss_weight import LossWeight

from torchinfo import summary

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
        self.num_time_features = 5

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

        self.enable_early_stopping = False
        self.early_stopping_delta = 0.01

        self.model_save = True

        self.freq = freq
        self.epochs = epochs
        self.num_parallel_samples = num_parallel_samples
        
        self.model = Estimator(
            input_size=self.input_length,
            output_size=self.output_length,
            prediction_length=prediction_length,
            context_length=context_length
        )

        summary(self.model, input_size=(64, target_dim, self.input_length))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.add_weight_loss = False

        if self.is_scaling and self.add_weight_loss:
            self.criterion = nn.GaussianNLLLoss(reduction="none")
        else:
            self.criterion = nn.GaussianNLLLoss(reduction="mean", eps=1e-6)

        #self.scaler = Scaler("abs_mean", self.feature_second)
        #self.scaler = Scaler("mean", self.feature_second)
        self.scaler = Scaler("standard", self.feature_second)

        self.path = "checkpoint.pth"

    def permute_dim(self, x):
        # 元の次元 [batch_size, feature_size, time_step]
        if self.feature_second:
            return x.permute(0, 2, 1)
        else:
            # 変換後 [batch_size, time_step, feature_size]
            return x.permute(0, 1, 2)

    def pre_prepare(self, input_x: torch.tensor, target_x: torch.tensor, time_features: torch.tensor, extention_features: torch.tensor):
        input_x = self.permute_dim(input_x).to(self.device)
        target_x = self.permute_dim(target_x).to(self.device)
        time_features = time_features.to(self.device)
        extention_features = extention_features.to(self.device)

        if self.add_weight_loss:
            self.loss_weight.calc_weight(input_x)

        if self.is_scaling:
            input_x, scale = self.scaler.fit_transform(input_x)
        else:
            scale = None

        if self.add_time_features:
            input_x = torch.cat([input_x, time_features], dim=2)

        if self.add_extention_features:
            input_x = torch.cat([input_x, extention_features], dim=2)

        return input_x, target_x, scale

    def loss_compute(self, mean: torch.tensor, target_x: torch.tensor, var: torch.tensor, scale: torch.tensor):
        if self.is_scaling:
            mean, var = self.scaler.invert_transform(mean, var, scale)

        if self.is_scaling and self.add_weight_loss:
            loss = self.criterion(mean, target_x, var)

            loss = self.loss_weight.add_weight(loss)
        else:
            loss = self.criterion(mean, target_x, var)

        return loss, mean, var

    def train(self, dataset: torch.utils.data.DataLoader, val_dataset: torch.utils.data.DataLoader = None, train_target: torch.tensor = None):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        train_loss = []
        val_loss = []

        minimal_val_loss = {"loss": np.inf, "epoch": 0}

        if train_target is not None:
            self.loss_weight = LossWeight(self.permute_dim(train_target).to(self.device))

        for epoch in range(self.epochs):
            self.model.train()

            train_loss_per_epoch = np.array([])

            for i, (start_date, input_x, target_x, time_features, extention_features) in enumerate(dataset):
                optimizer.zero_grad()
                batch_size = input_x.shape[0]
                hidden = self.model.init_hidden(batch_size)

                input_x, target_x, scale = self.pre_prepare(input_x, target_x, time_features, extention_features)
                
                mean, var = self.model(input_x, hidden)

                loss, mean, var = self.loss_compute(mean, target_x, var, scale)

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

                        input_x, target_x, scale = self.pre_prepare(input_x, target_x, time_features, extention_features)
                
                        with torch.no_grad():
                            mean, var = self.model(input_x, hidden)

                        loss, mean, var = self.loss_compute(mean, target_x, var, scale)

                        val_loss_per_epoch = np.append(val_loss_per_epoch, loss.item())
            
                loss_mean = np.mean(val_loss_per_epoch)
                val_loss.append(loss_mean)

                if loss_mean < minimal_val_loss["loss"]:
                    minimal_val_loss["loss"] = loss_mean
                    minimal_val_loss["epoch"] = epoch

                    self.save()
                else:
                    if self.enable_early_stopping:
                        if loss_mean - minimal_val_loss["loss"] > self.early_stopping_delta:
                            print(f"early stopping: epoch {epoch+1}")
                            break

                print(f'Epoch [{epoch+1}/{self.epochs}], train_Loss: {train_loss[-1]:.4f}, val_loss: {val_loss[-1]:.4f}')
            else:
                print(f'Epoch [{epoch+1}/{self.epochs}], train_Loss: {train_loss[-1]:.4f}')

        if self.model_save:
            self.load()
            print(f"Load minimal model, epoch: {minimal_val_loss['epoch']}, loss: {minimal_val_loss['loss']}")
        else:
            self.model.eval()

        return train_loss, val_loss, minimal_val_loss

    def forecast(self, input_x: torch.tensor, time_features: torch.tensor, extention_features: torch.tensor):
        self.model.eval()

        input_x, target_x, scale = self.pre_prepare(input_x, input_x, time_features, extention_features)

        with torch.no_grad():
            mean, var = self.model(input_x)

        if self.is_scaling:
            mean, var = self.invert_scaling(mean, var, scale)

        mean = mean.cpu().numpy()
        var = var.cpu().numpy()

        output = ForecastOutput(mean, var, self.num_parallel_samples)
        
        return output

    def make_evaluation_predictions(self, input_x: torch.tensor, target_x: torch.tensor, time_features: torch.tensor, extention_features: torch.tensor):
        self.model.eval()

        input_x, target_x, scale = self.pre_prepare(input_x, target_x, time_features, extention_features)

        with torch.no_grad():
            mean, var = self.model(input_x)

        loss, mean, var = self.loss_compute(mean, target_x, var, scale)

        mean = self.permute_dim(mean).squeeze(0).permute(1, 0).cpu().numpy()
        var = self.permute_dim(var).squeeze(0).permute(1, 0).cpu().numpy()
        loss = loss.cpu().numpy()

        output = []
        for i in range(len(mean)):
            output.append(ForecastOutput(mean[i], var[i], self.num_parallel_samples))

        return output, loss

    def save(self):
        torch.save(self.model.state_dict(), self.path)

    def load(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        self.model.to(self.device)
    
        


