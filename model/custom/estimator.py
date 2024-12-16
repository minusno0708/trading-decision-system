import torch
import torch.nn as nn

class Estimator(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        prediction_length,
        context_length,
        hidden_size=64, 
        num_layers=2, 
        dropout_rate=0.2
    ):
        super(Estimator, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_rate,
                            batch_first=True)
        self.output_mu = nn.Linear(hidden_size, output_size)
        self.output_sigma = nn.Linear(hidden_size, output_size)
        self.prediction_length = prediction_length
        self.context_length = context_length

    def forward(self, x, hidden_state=None):
        lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)
        
        mean = self.output_mu(lstm_out)
        var = torch.exp(self.output_sigma(lstm_out))  # ensuring sigma is positive

        if self.prediction_length < self.context_length:
            mean = mean[:, -self.prediction_length:, :]
            var = var[:, -self.prediction_length:, :]
        
        if self.prediction_length == 1:
            mean = mean[:, -1, :].unsqueeze(1)
            var = var[:, -1, :].unsqueeze(1)

        return mean, var

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden

