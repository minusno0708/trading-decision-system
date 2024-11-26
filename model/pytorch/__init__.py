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

        
        self.model = Estimator(
            context_length=context_length,
            prediction_length=prediction_length,
            freq=freq,
            epochs=epochs,
            num_parallel_samples=num_parallel_samples,
            model_type=model_type
        )
        

    def train(self, dataset: pd.DataFrame):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()

        for epoch in range(self.epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()  # Zero out gradients
                batch_size = inputs.shape[0]
                hidden = self.model.init_hidden(batch_size)  # Initialize hidden state
                
                # Forward pass
                mu, sigma, _ = self.model(inputs, hidden)
                
                # Calculate loss
                loss = criterion(mu, targets) + torch.mean(torch.log(sigma))
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

    def forecast(self, dataset: pd.DataFrame):
        return list(self.model.predict(dataset))

