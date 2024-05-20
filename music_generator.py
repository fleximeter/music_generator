"""
File: music_generator.py

Featurizes a music21 staff for running through LSTM
"""

import torch
import torch.nn as nn

class LSTMMusic(nn.Module):
    """
    A class for making music LSTM models. It expects 3D tensors for prediction.
    Dimension 1 size: Batch size
    Dimension 2 size: Sequence length
    Dimension 3 size: Number of features
    """
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=1):
        """
        Initializes the music LSTM
        :param input_size: The input size
        :param output_size: The number of output categories
        :param hidden_size: The size of the hidden state vector
        :param num_layers: The number of layers to use
        """
        super(LSTMMusic, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x, hidden_states):
        output, hidden_states = self.lstm(x, hidden_states)
        output = self.output_layer(output[:, -1, :])
        return output, hidden_states
    
    def init_hidden(self, batch_size=1, device="cpu"):
        """
        Initializes the hidden state
        :param batch_size: The batch size
        :param device: The device (cpu, cuda, mps)
        """
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device), torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))
