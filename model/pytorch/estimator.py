import torch
import torch.nn as nn

class Estimator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(Estimator, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_rate,
                            batch_first=True)
        self.output_mu = nn.Linear(hidden_size, output_size)
        self.output_sigma = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        # LSTM forward pass
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        
        # Predicting the mean (mu) and standard deviation (sigma)
        mu = self.output_mu(lstm_out)
        sigma = torch.exp(self.output_sigma(lstm_out))  # ensuring sigma is positive

        return mu, sigma, hidden_state

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden

