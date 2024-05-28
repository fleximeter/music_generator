"""
File: dataset.py

This module has a dataset class for storing sequences.
"""

import featurizer
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple


class MusicXMLDataSet(Dataset):
    """
    Makes a dataset of sequenced notes based on a music XML corpus. This dataset
    will make sequences of notes and labels for the next note in the sequence,
    for generative training. It will exhaustively make sequences between a specified
    minimum sequence length and maximum sequence length, and these sequences should
    be provided to a DataLoader in shuffled fashion. Because the sequence lengths
    vary, it is necessary to provide a collate function to the DataLoader, and a
    collate function is provided as a static function in this class.
    """
    def __init__(self, score_list, min_sequence_length, max_sequence_length, label_keys) -> None:
        """
        Makes a MusicXMLDataSet
        :param score_list: A list of music21 scores to turn into a dataset
        :param min_sequence_length: The minimum sequence length
        :param max_sequence_length: The maximum sequence length
        :param label_keys: A list of label keys. Must match one of several predefined key sets
        accepted by the featurizer.make_labels() function.
        """
        super(MusicXMLDataSet, self).__init__()
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.data, self.labels = self._load_data(score_list, label_keys)
        
    def __len__(self) -> int:
        """
        Gets the number of entries in the dataset
        :return: The number of entries in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Gets the next item and its labels in the dataset
        :return: sample, labels
        """
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, *label
    
    def _load_data(self, score_list, label_keys) -> Tuple[list, list]:
        """
        Parses each MusicXML file and generates sequences and labels from it
        :param score_list: A list of MusicXML files to turn into a dataset
        :param label_keys: A list of label keys. Must match one of several predefined key sets
        accepted by the featurizer.make_labels() function.
        """
        sequences = []
        labels = []
        for score in score_list:
            # Go through each staff in each score, and generate individual
            # sequences and labels for that staff
            for i in featurizer.get_staff_indices(score):
                data = featurizer.load_data(score[i])
                data = featurizer.make_one_hot_features(data, False)
                for j in range(self.min_sequence_length, self.max_sequence_length + 1):
                    seq = featurizer.make_n_gram_sequences(data, j+1)
                    lab = featurizer.make_labels(seq, label_keys)

                    # trim the last entry off the sequence, because it is the label
                    sequences += [s[:-1, :] for s in seq]
                    labels += lab

        return sequences, labels
    
    def collate(batch):
        """
        Pads a batch in preparation for training. This is necessary
        because we expect the dataloader to randomize the data, which will
        mix sequences of different lengths. We will pad these sequences with empty
        entries so we can run everything through the model at the same time.
        :param batch: A batch produced by a DataLoader
        :return: The padded sequences, labels, and sequence lengths
        """
        # Sort the batch in order of sequence length. This is required by the pack_padded_sequences function. 
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        zipped_batch = list(zip(*batch))
        sequences = zipped_batch[0]
        targets = zipped_batch[1:]
        lengths = torch.tensor([seq.shape[0] for seq in sequences])
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        targets = [torch.tensor(target) for target in targets]
        return sequences_padded, *targets, lengths
    
    def prepare_prediction(sequence, max_length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares a sequence for prediction. This function does the padding process
        just like the collate function, so the model behaves as expected.
        :param sequence: The sequence to prepare
        :param max_length: The maximum sequence length the model was trained on
        :return: The padded sequence and a list of lengths
        """
        lengths = torch.tensor([sequence.shape[1]])
        if sequence.shape[1] < max_length:
            zeros = torch.zeros((1, max_length - sequence.shape[1], sequence.shape[2]))
            sequence = torch.cat((sequence, zeros), dim=1)
        return sequence, lengths
    