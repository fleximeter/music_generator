"""
File: model_definition.py

This file contains the neural network definition for the music sequence generator. 
At this point it uses a LSTM model. It can take a variable number of output label 
classes - you define the output sizes when initializing the model.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple

class LSTMMusic(nn.Module):
    """
    A class for making music LSTM models. It expects 3D tensors for prediction.
    Dimension 1 size: Batch size
    Dimension 2 size: Sequence length
    Dimension 3 size: Number of features
    
    There are three outputs:
    y1: letter, accidental name logits
    y2: octave logits
    y3: quarter length logits
    """
    def __init__(self, input_size, output_sizes, hidden_size=128, num_layers=1, device="cpu"):
        """
        Initializes the music LSTM
        :param input_size: The input size
        :param output_sizes: A list of output sizes. A separate output layer will be created for each entry in the list.
        :param hidden_size: The size of the hidden state vector
        :param num_layers: The number of layers to use
        :param device: The device that the model will run on
        """
        super(LSTMMusic, self).__init__()

        # The layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layers = [nn.Linear(hidden_size, output_size) for output_size in output_sizes]
        
        # Track information about the model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        # Use He initialization to help avoid vanishing gradients
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

    def forward(self, x, lengths, hidden_states) -> Tuple[list, tuple]:
        """
        Runs a batch of sequences forward through the model
        :param x: The batch of sequences
        :param lengths: A Tensor with sequence lengths for the corresponding sequences
        :param hidden_states: A tuple of hidden state matrices
        :return (y1, y2, y3), hidden: Returns a logit tuple
        [output logits] and updated hidden states
        """
        # pack the input, run it through the model, and unpack it
        packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        packed_output, hidden_states = self.lstm(packed_input, hidden_states)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # get the index of the last output
        idx = (lengths - 1).view(-1, 1, 1).expand(output.size(0), 1, output.size(2)).to(self.device)
        last_output = output.gather(1, idx).squeeze(1)
        
        # run the LSTM output through the final layers to generate the logits
        output_logits = [output(last_output) for output in self.output_layers]
        return output_logits, hidden_states
    
    def init_hidden(self, batch_size=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden state
        :param batch_size: The batch size
        :return: Returns a tuple of empty hidden matrices
        """
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device), 
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device))
    