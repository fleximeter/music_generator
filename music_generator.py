"""
File: music_generator.py

This file contains the neural network definition for the music sequence
generator. At this point it uses a LSTM model and outputs three labels:
pitch class, octave, and quarter length.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMMusic(nn.Module):
    """
    A class for making music LSTM models. It expects 3D tensors for prediction.
    Dimension 1 size: Batch size
    Dimension 2 size: Sequence length
    Dimension 3 size: Number of features
    
    There are three outputs:
    y1: pitch class logits
    y2: octave logits
    y3: quarter length logits
    """
    def __init__(self, input_size, output_size_pc, output_size_octave, output_size_quarter_length, hidden_size=128, num_layers=1, device="cpu"):
        """
        Initializes the music LSTM
        :param input_size: The input size
        :param output_size_pc: The number of output categories for pitch class
        :param output_size_octave: The number of output categories for octave
        :param output_size_quarter_length: The number of output categories for quarter length
        :param hidden_size: The size of the hidden state vector
        :param num_layers: The number of layers to use
        """
        super(LSTMMusic, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # The output layers
        self.output_pc = nn.Linear(hidden_size, output_size_pc)
        self.output_octave = nn.Linear(hidden_size, output_size_octave)
        self.output_quarter_length = nn.Linear(hidden_size, output_size_quarter_length)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, lengths, hidden_states):
        """
        Runs a batch of sequences forward through the model
        :param x: The batch of sequences
        :param lengths: A Tensor with sequence lengths for the corresponding sequences
        :param hidden_states: A tuple of hidden state matrices
        :return (y1, y2, y3), hidden: Returns a logit tuple
        (pitch class logits, octave logits, and quarter length logits) and updated hidden states
        """
        # pack the input, run it through the model, and unpack it
        packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        packed_output, hidden_states = self.lstm(packed_input, hidden_states)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # get the index of the last output
        idx = (lengths - 1).view(-1, 1, 1).expand(output.size(0), 1, output.size(2)).to(self.device)
        last_output = output.gather(1, idx).squeeze(1)
        
        # run the LSTM output through the final layers to generate the logits
        output_pc = self.output_pc(last_output)
        output_octave = self.output_octave(last_output)
        output_quarter_length = self.output_quarter_length(last_output)
        return (output_pc, output_octave, output_quarter_length), hidden_states
    
    def init_hidden(self, batch_size=1):
        """
        Initializes the hidden state
        :param batch_size: The batch size
        :return: Returns a tuple of empty hidden matrices
        """
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device), torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device))
    