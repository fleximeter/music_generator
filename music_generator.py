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
    y1: letter, accidental name logits
    y2: octave logits
    y3: quarter length logits
    """
    def __init__(self, input_size, output_sizes, hidden_size=128, num_layers=1, device="cpu"):
        """
        Initializes the music LSTM
        :param input_size: The input size
        :param output_size_letter_name: The number of output categories for note letter name
        :param output_size_accidental_name: The number of output categories for note accidental name
        :param output_size_octave: The number of output categories for octave
        :param output_size_quarter_length: The number of output categories for quarter length
        :param hidden_size: The size of the hidden state vector
        :param num_layers: The number of layers to use
        """
        super(LSTMMusic, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # The output layers
        self.output_letter_accidental_name = nn.Linear(hidden_size, output_sizes[0])
        self.output_octave = nn.Linear(hidden_size, output_sizes[1])
        self.output_quarter_length = nn.Linear(hidden_size, output_sizes[2])
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        # Use He initialization to help avoid vanishing gradients
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

    def forward(self, x, lengths, hidden_states):
        """
        Runs a batch of sequences forward through the model
        :param x: The batch of sequences
        :param lengths: A Tensor with sequence lengths for the corresponding sequences
        :param hidden_states: A tuple of hidden state matrices
        :return (y1, y2, y3), hidden: Returns a logit tuple
        (letter accidental name logits, octave logits, quarter length logits) and updated hidden states
        """
        # pack the input, run it through the model, and unpack it
        packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        packed_output, hidden_states = self.lstm(packed_input, hidden_states)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # get the index of the last output
        idx = (lengths - 1).view(-1, 1, 1).expand(output.size(0), 1, output.size(2)).to(self.device)
        last_output = output.gather(1, idx).squeeze(1)
        
        # run the LSTM output through the final layers to generate the logits
        letter_accidental_name_logits = self.output_letter_accidental_name(last_output)
        octave_logits = self.output_octave(last_output)
        quarter_length_logits = self.output_quarter_length(last_output)
        return (letter_accidental_name_logits, octave_logits, quarter_length_logits), hidden_states
    
    def init_hidden(self, batch_size=1):
        """
        Initializes the hidden state
        :param batch_size: The batch size
        :return: Returns a tuple of empty hidden matrices
        """
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device), 
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device))
    