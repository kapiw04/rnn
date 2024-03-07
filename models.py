"""
This file contains the model for the project.
"""
from hyperparameters import DEVICE
from torch import nn, zeros

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate) -> None:
        """
            input size: The character vocabulary size
            hidden_size: The number of hidden units in the RNN
            num_layers: The number of layers in the RNN
            dropout_rate: The dropout rate for the RNN
        """
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True, device=DEVICE)
        # Size of input should look like this (batch_size, seq_length, input_size) -> (64, 100, VOCAB_SIZE)
        # If not in batch the input should look like this (seq_length, input_size) -> (1, 100, VOCAB_SIZE)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        """
            x: The input to the RNN
        """
        h0 = self.init_hidden(x.shape[0])
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


    
    def init_hidden(self, batch_size):
        """
            batch_size: The batch size
        """
        # The hidden state should be of size (num_layers, batch_size, hidden_size)
        return zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
    