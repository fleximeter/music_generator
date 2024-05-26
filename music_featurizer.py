"""
File: music_featurizer.py

This module has featurization functions for preparing data to run through the model,
and for postprocessing generated data. It also has a dataset class for storing
sequences.
"""

import music21
import torch
import torch.nn.functional as F
import math
import xml_gen
import numpy as np
from fractions import Fraction
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple


###################################################################################################################
# Feature dictionaries
###################################################################################################################

_LETTER_NAME_ENCODING = {"None": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "A": 6, "B": 7}
_ACCIDENTAL_ENCODING = {"None": 0, "-2.0": 1, "-1.0": 2, "0.0": 3, "1.0": 4, "2.0": 5}
_ACCIDENTAL_NAME_ENCODING = {"None": 0, 'double-flat': 1, 'double-sharp': 2, 'flat': 3, 'half-flat': 4, 'half-sharp': 5, 
                             'natural': 6, 'one-and-a-half-flat': 7, 'one-and-a-half-sharp': 8, 'quadruple-flat': 9, 
                             'quadruple-sharp': 10, 'sharp': 11, 'triple-flat': 12, 'triple-sharp': 13}
_PITCH_SPACE_ENCODING = {"None": 0}
_OCTAVE_ENCODING = {"None": 0}
_PITCH_CLASS_ENCODING = {"None": 0}
_KEY_SIGNATURE_ENCODING = {"None": 0}
_MODE_ENCODING = {"None": 0, "major": 1, "minor": 2}
_BEAT_ENCODING = {"None": 0}
_QUARTER_LENGTH_ENCODING = {"None": 0}
_PITCH_SPACE_REVERSE_ENCODING = {0: "None"}
_LETTER_NAME_REVERSE_ENCODING = {0: "None", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "A", 7: "B"}
_ACCIDENTAL_NAME_REVERSE_ENCODING = {0: "None", 1: 'double-flat', 2: 'double-sharp', 3: 'flat', 4: 'half-flat', 
                                     5: 'half-sharp', 6: 'natural', 7: 'one-and-a-half-flat', 8: 'one-and-a-half-sharp', 
                                     9: 'quadruple-flat', 10: 'quadruple-sharp', 11: 'sharp', 12: 'triple-flat', 13: 'triple-sharp'}
_OCTAVE_REVERSE_ENCODING = {0: "None"}
_PITCH_CLASS_REVERSE_ENCODING = {0: "None"}
_KEY_SIGNATURE_REVERSE_ENCODING = {0: "None"}
_MODE_REVERSE_ENCODING = {0: "None", 1: "major", 2: "minor"}
_BEAT_REVERSE_ENCODING = {0: "None"}
_QUARTER_LENGTH_REVERSE_ENCODING = {0: "None"}

# Lets you convert accidental names to chromatic alteration
_ACCIDENTAL_NAME_TO_ALTER_ENCODING = {"None": 0.0, 'double-flat': 2.0, 'double-sharp': -2.0, 'flat': 1.0, 'half-flat': 0.5, 'half-sharp': -0.5, 
                                     'natural': 0.0, 'one-and-a-half-flat': 1.5, 'one-and-a-half-sharp': -1.5, 'quadruple-flat': 4.0, 
                                     'quadruple-sharp': -4.0, 'sharp': -1.0, 'triple-flat': 3.0, 'triple-sharp': -3.0}

_ACCIDENTAL_ALTER_TO_NAME_ENCODING = {0.0: "None", 2.0: 'double-flat', -2.0: 'double-sharp', 1.0: 'flat', 0.5: 'half-flat', -0.5: 'half-sharp', 
                                      1.5: 'one-and-a-half-flat', -1.5: 'one-and-a-half-sharp', 4.0: 'quadruple-flat', 
                                      -4.0: 'quadruple-sharp', -1.0: 'sharp', 3.0: 'triple-flat', -3.0: 'triple-sharp'}

##########################################################################
# Generate pitch encoding
##########################################################################

for i in range(1, 128+1):
    _PITCH_SPACE_ENCODING[str(float(i-1))] = i 
    _PITCH_SPACE_REVERSE_ENCODING[i] = str(float(i-1))

for i in range(1, 14+1):
    _OCTAVE_ENCODING[str(i-1)] = i
    _OCTAVE_REVERSE_ENCODING[i] = str(i-1)

for i in range(1, 12+1):
    _PITCH_CLASS_ENCODING[str(float(i-1))] = i
    _PITCH_CLASS_REVERSE_ENCODING[i] = str(float(i-1))

for i in range(1, 128+1):
    _PITCH_SPACE_ENCODING[str(float(i-1))] = i 
    _PITCH_SPACE_REVERSE_ENCODING[i] = str(float(i-1))

for i in range(1, 15+1):
    _KEY_SIGNATURE_ENCODING[str(i-8)] = i 
    _KEY_SIGNATURE_REVERSE_ENCODING[i] = str(i-8)

##########################################################################
# Generate beat and quarter length encoding
##########################################################################

# This sets the maximum note duration in quarter notes that the model can handle.
_MAX_QUARTER_LENGTH = 8

idx_quarter_length = 1

# Quarters
for i in range(1, _MAX_QUARTER_LENGTH):
    _QUARTER_LENGTH_ENCODING[f"{i}"] = idx_quarter_length
    _QUARTER_LENGTH_REVERSE_ENCODING[idx_quarter_length] = f"{i}"
    _BEAT_ENCODING[f"{i}"] = idx_quarter_length
    _BEAT_REVERSE_ENCODING[idx_quarter_length] = f"{i}"
    idx_quarter_length += 1

# 8ths, triplet 8ths, 16ths, triplet 16ths, 32nds. The first value
# in the tuple is the quarter length denominator, and the second value
# is a step value. The step value helps to avoid duplicate duration
# values in the encoding. In general it should probably be 2, except 
# for triplet eighths.
for denominator, step in [(2, 2), (3, 1), (4, 2), (6, 2), (8, 2)]:
    for i in range(1, _MAX_QUARTER_LENGTH * denominator, step):
        # This condition catches duplicate durations for subdivisions of 3 and 6
        if denominator % 3 != 0 or i % 3 != 0:
            _QUARTER_LENGTH_ENCODING[f"{i}/{denominator}"] = idx_quarter_length
            _QUARTER_LENGTH_REVERSE_ENCODING[idx_quarter_length] = f"{i}/{denominator}"
            _BEAT_ENCODING[f"{i}/{denominator}"] = idx_quarter_length
            _BEAT_REVERSE_ENCODING[idx_quarter_length] = f"{i}/{denominator}"
            idx_quarter_length += 1


###################################################################################################################
# The total number of features and outputs for the model. This can change from time to time, and must be updated!
###################################################################################################################
_NUM_FEATURES = len(_LETTER_NAME_ENCODING) + len(_ACCIDENTAL_NAME_ENCODING) + len(_OCTAVE_ENCODING) + len(_QUARTER_LENGTH_ENCODING) + \
                len(_BEAT_ENCODING) + len(_PITCH_CLASS_ENCODING)  + len(_KEY_SIGNATURE_ENCODING)  + len(_MODE_ENCODING) 
_NUM_OUTPUTS = len(_LETTER_NAME_ENCODING) + len(_ACCIDENTAL_NAME_ENCODING) + len(_OCTAVE_ENCODING) + len(_QUARTER_LENGTH_ENCODING)


###################################################################################################################
# Functions for featurization
###################################################################################################################

def calculate_next_beat(note) -> Fraction:
    """
    Calculates the beat of the next note based on the time signature and beat length
    of the current note
    :param note: The note dictionary to calculate from
    :return: The beat of the next note
    """
    time_signature_numerator = Fraction(note["time_signature"].split("/")[0])
    beat = Fraction(note["beat"]) + Fraction(note["quarterLength"])
    while beat > time_signature_numerator:
        beat -= time_signature_numerator
    return beat


def convert_letter_accidental_octave_to_note(letter_name, accidental_name, octave) -> dict:
    """
    Gets the following: pitch class, octave, pitch space value, letter name, and accidental
    from provided letter name, accidental, and octave strings
    :param letter_name: The letter name string
    :param accidental_name: The accidental string
    :param octave: The octave string
    :return: A note dictionary {pitch class, octave, pitch space value, letter name, accidental}
    """
    PC_MAP = {'C': 0.0, 'D': 2.0, 'E': 4.0, 'F': 5.0, 'G': 7.0, 'A': 9.0, 'B': 11.0}
    note = {}

    if letter_name != "None" and octave != "None":
        note["octave"] = int(float(octave))
        note["letter_name"] = letter_name
        note["accidental_name"] = accidental_name
        note["accidental"] = _ACCIDENTAL_NAME_TO_ALTER_ENCODING[accidental_name]
        note["pitch_class_id"] = (PC_MAP[letter_name] + _ACCIDENTAL_NAME_TO_ALTER_ENCODING[accidental_name]) % 12
        note["ps"] = note["pitch_class_id"] + (note["octave"] + 1) * 12
    else:
        note["octave"] = "None"
        note["pitch_class_id"] = "None"
        note["ps"] = "None"
        note["letter_name"] = "None"
        note["accidental_name"] = "None"
        note["accidental"] = "None"

    return note


def convert_pc_octave_to_note(pc, octave) -> dict:
    """
    Gets the following: pitch class, octave, pitch space value, letter name, and accidental
    from provided pitch class and octave strings
    :param pc: The pitch class string
    :param octave: The octave string
    :return: A note dictionary {pitch class, octave, pitch space value, letter name, accidental}
    """
    PC_MAP = [('C', 0), ('C', -1), ('D', 0), ('D', -1), ('E', 0), ('F', 0), ('F', -1), ('G', 0), ('G', -1), ('A', 0), ('A', -1), ('B', 0)]
    # PC_MAP = {'0.0': 'C', '2.0': 'D', '4.0': 'E', '5.0': 'F', '7.0': 'G', '9.0': 'A', '11.0': 'B'}
    note = {}

    if pc != "None" and octave != "None":
        note["octave"] = int(float(octave))
        note["pitch_class_id"] = float(pc)
        note["ps"] = note["pitch_class_id"] + (note["octave"] + 1) * 12
        microtone, semitone = math.modf(note["ps"])
        pc = semitone % 12
        note["letter_name"], accidental_alter = PC_MAP[int(pc)]
        accidental_alter -= microtone
        note["accidental"] = accidental_alter
        note["accidental_name"] = _ACCIDENTAL_ALTER_TO_NAME_ENCODING[accidental_alter]
    else:
        note["octave"] = "None"
        note["pitch_class_id"] = "None"
        note["ps"] = "None"
        note["letter_name"] = "None"
        note["accidental"] = "None"
        note["accidental_name"] = "None"

    return note    


def get_staff_indices(score) -> list:
    """
    Identifies the staff indices in a music21 score, since not all
    entries in the score are staves.
    :param score: The music21 score
    :return: A list of staff indices
    """
    indices = []
    for i, item in enumerate(score):
        if type(item) == music21.stream.Part or type(item) == music21.stream.PartStaff:
            indices.append(i)
    return indices


def load_data(staff) -> list:
    """
    Loads a Music21 staff and featurizes it
    :param staff: The staff to load
    :return: The tokenized score as a list of note dictionaries 
    """
    dataset = []
    tie_status = False
    current_note = {}
    current_time_signature = "4/4"
    current_key = 0
    current_mode = "major"

    # We assume there are not multiple voices on this staff, and there are no chords - it's just a line
    for measure in staff:
        if type(measure) == music21.stream.Measure:
            for item in measure:
                if type(item) == music21.meter.TimeSignature:
                    current_time_signature = item.ratioString
                elif type(item) == music21.key.Key:
                    current_key = item.sharps
                    current_mode = item.mode
                elif type(item) == music21.note.Note:
                    if not tie_status:
                        current_note["ps"] = item.pitch.ps                                     # MIDI number (symbolic x257)
                        current_note["octave"] = item.pitch.octave                             # Octave number (symbolic)
                        current_note["letter_name"] = item.pitch.step                          # letter name (C, D, E, ...) (symbolic x8)

                        # accidental (symbolic)
                        current_note["accidental_name"] = item.pitch.accidental.name if item.pitch.accidental is not None else "None"
                        current_note["accidental"] = item.pitch.accidental.alter if item.pitch.accidental is not None else 0.0
                        current_note["pitch_class_id"] = float(item.pitch.pitchClass)          # pitch class number 0, 1, 2, ... 11 (symbolic x13)
                        current_note["key_signature"] = current_key                            # key signature
                        current_note["mode"] = current_mode                                    # mode

                        current_note["quarterLength"] = item.duration.quarterLength            # duration in quarter notes (symbolic)
                        current_note["beat"] = item.beat                                       # beat (symbolic)
                        current_note["time_signature"] = current_time_signature                # time signature (symbolic)

                        if item.tie is not None and item.tie.type in ["continue", "stop"]:
                            tie_status = True
                        else:
                            dataset.append(current_note)
                            current_note = {}
                        
                    else:
                        current_note["quarterLength"] += item.duration.quarterLength
                        if item.tie is None or item.tie.type != "continue":
                            tie_status = False
                            dataset.append(current_note)
                            current_note = {}

                elif type(item) == music21.note.Rest:
                    tie_status = False
                    current_note["ps"] = "None"                                  # MIDI number (symbolic x257)
                    current_note["octave"] = "None"                              # MIDI number (symbolic x257)
                    current_note["letter_name"] = "None"                         # letter name (C, D, E, ...) (symbolic x8)
                    current_note["accidental_name"] = "None"                     # accidental name ("sharp", etc.) (symbolic)
                    current_note["accidental"] = "None"                          # accidental alter value (symbolic)
                    current_note["pitch_class_id"] = "None"                      # pitch class number 0, 1, 2, ... 11 (symbolic x13)
                    current_note["key_signature"] = current_key                  # key signature
                    current_note["mode"] = current_mode                          # mode
                    current_note["quarterLength"] = item.duration.quarterLength  # duration in quarter notes (symbolic)
                    current_note["beat"] = item.beat                             # beat in quarter notes (symbolic)
                    current_note["time_signature"] = current_time_signature      # time signature (symbolic)
                    dataset.append(current_note)
                    current_note = {}

    return dataset


def make_labels(x) -> list:
    """
    Generates a label list for a list of sequences (2D tensors). The label is
    calculated for a particular index in dimension 1 of the 2D tensors.
    Dimension 1 is the batch length
    Dimension 2 is the total sequence length
    Dimension 3 is the features of individual notes in the sequence
    :param x: A list of sequences
    :return: A list of label tuples. Each label tuple has 4 labels (letter name, accidental name, octave, quarter length).
    """
    y = []
    for sequence in x:
        i = (
            (0, len(_LETTER_NAME_ENCODING)), 
            (len(_LETTER_NAME_ENCODING), len(_LETTER_NAME_ENCODING) + len(_ACCIDENTAL_NAME_ENCODING)),
            (len(_LETTER_NAME_ENCODING) + len(_ACCIDENTAL_NAME_ENCODING), len(_LETTER_NAME_ENCODING) + len(_ACCIDENTAL_NAME_ENCODING) + len(_OCTAVE_ENCODING)),
            (len(_LETTER_NAME_ENCODING) + len(_ACCIDENTAL_NAME_ENCODING) + len(_OCTAVE_ENCODING), len(_LETTER_NAME_ENCODING) + len(_ACCIDENTAL_NAME_ENCODING) + len(_OCTAVE_ENCODING) + len(_QUARTER_LENGTH_ENCODING))
        )
        letter_name = sequence[-1, i[0][0]:i[0][1]]
        accidental_name = sequence[-1, i[1][0]:i[1][1]]
        octave = sequence[-1, i[2][0]:i[2][1]]
        quarter_length = sequence[-1, i[3][0]:i[3][1]]
        y.append((letter_name.argmax().item(), accidental_name.argmax().item(), octave.argmax().item(), quarter_length.argmax().item()))
    return y


def make_n_gram_sequences(tokenized_dataset, n) -> list:
    """
    Makes N-gram sequences from a tokenized dataset
    :param tokenized_dataset: The tokenized dataset
    :param n: The length of the n-grams
    :return: X
    X is a list of N-gram tensors
      (dimension 1 is the entry in the N-gram)
      (dimension 2 has the features of the entry)
    """
    x = []
    for j in range(n, tokenized_dataset.shape[0] - 1):
        y = []
        for k in range(j-n, j):
            y.append(tokenized_dataset[k, :])
        x.append(torch.vstack(y))
    return x


def make_one_hot_features(dataset: list, batched=True) -> torch.Tensor:
    """
    Turns a dataset into a list of one-hot-featured instances in preparation for 
    running it through a model. You can use this for making predictions if you want.
    :param dataset: The dataset to make one-hot
    :param batched: Whether or not the data is expected to be in batched format (3D) 
    or unbatched format (2D). If you will be piping this data into the make_sequences 
    function, it should not be batched. In all other cases, it should be batched.
    :return: The one-hot data as a 2D or 3D tensor
    """
    instances = []
    for instance in dataset:
        # One-hots
        pitch_class_one_hot = F.one_hot(torch.tensor(_PITCH_CLASS_ENCODING[str(instance["pitch_class_id"])]), len(_PITCH_CLASS_ENCODING)).float()
        octave_one_hot = F.one_hot(torch.tensor(_OCTAVE_ENCODING[str(instance["octave"])]), len(_OCTAVE_ENCODING)).float()
        if instance["quarterLength"] == "None":
            quarter_length_one_hot = F.one_hot(torch.tensor(_QUARTER_LENGTH_ENCODING[str(instance["quarterLength"])]), len(_QUARTER_LENGTH_ENCODING)).float()
        else:
            quarter_length_one_hot = F.one_hot(torch.tensor(_QUARTER_LENGTH_ENCODING[str(Fraction(instance["quarterLength"]))]), len(_QUARTER_LENGTH_ENCODING)).float()
        if instance["beat"] == "None":
            beat_one_hot = F.one_hot(torch.tensor(_BEAT_ENCODING[str(instance["beat"])]), len(_BEAT_ENCODING)).float()
        else:
            beat_one_hot = F.one_hot(torch.tensor(_BEAT_ENCODING[str(Fraction(instance["beat"]))]), len(_BEAT_ENCODING)).float()
        letter_name_one_hot = F.one_hot(torch.tensor(_LETTER_NAME_ENCODING[str(instance["letter_name"])]), len(_LETTER_NAME_ENCODING)).float()
        accidental_name_one_hot = F.one_hot(torch.tensor(_ACCIDENTAL_NAME_ENCODING[str(instance["accidental_name"])]), len(_ACCIDENTAL_NAME_ENCODING)).float()
        key_signature_one_hot = F.one_hot(torch.tensor(_KEY_SIGNATURE_ENCODING[str(instance["key_signature"])]), len(_KEY_SIGNATURE_ENCODING)).float()
        mode_one_hot = F.one_hot(torch.tensor(_MODE_ENCODING[str(instance["mode"])]), len(_MODE_ENCODING)).float()
        instances.append(torch.hstack((letter_name_one_hot, accidental_name_one_hot, octave_one_hot, quarter_length_one_hot, beat_one_hot, 
                                       pitch_class_one_hot, key_signature_one_hot, mode_one_hot)))

    instances = torch.vstack(instances)
    if batched:
        instances = torch.reshape(instances, (1,) + instances.shape)
    return instances


def retrieve_class_dictionary(prediction: tuple) -> dict:
    """
    Retrives a predicted class's information based on its id
    :param prediction: The prediction tuple
    :return: The prediction dictionary
    """
    note = {"quarterLength": Fraction(_QUARTER_LENGTH_REVERSE_ENCODING[prediction[3]])}
    note.update(convert_letter_accidental_octave_to_note(_LETTER_NAME_REVERSE_ENCODING[prediction[0]], _ACCIDENTAL_NAME_REVERSE_ENCODING[prediction[1]], _OCTAVE_REVERSE_ENCODING[prediction[2]]))
    return note


def unload_data(dataset: list) -> music21.stream.Score:
    """
    Unloads data and turns it into a score again, in preparation for
    rendering a MusicXML file.
    :param dataset: The dataset to unload
    :param time_signature: The time signature to use
    :return: A music21 score
    """
    MAX_MEASURE_NUMBERS = 50
    notes = []
    rhythms = []
    time_signature = "4/4"
    key_signature = 0
    padding_left_first_measure = 0
    if len(dataset) > 0:
        time_signature = dataset[0]["time_signature"]
        key_signature = dataset[0]["key_signature"]
        padding_left_first_measure = dataset[0]["beat"] - 1
    for item in dataset:
        if item["letter_name"] == "None":
            notes.append(-np.inf)
        else:
            x = (item["letter_name"], item["octave"], item["accidental_name"])
            notes.append((item["letter_name"], item["octave"], item["accidental_name"]))
        rhythms.append(float(item["quarterLength"]))
    notes_m21 = xml_gen.make_music21_list(notes, rhythms)
    score = xml_gen.create_score()
    xml_gen.add_instrument(score, "Cello", "Vc.")
    if len(notes) > 0:
        ts = [int(t) for t in time_signature.split('/')]
        bar_duration = ts[0] * 4 / ts[1]
        first_measure_num = 0 if padding_left_first_measure > 0.0 else 1
        xml_gen.add_measures(score, MAX_MEASURE_NUMBERS, first_measure_num, key_signature, time_signature, bar_duration, 0.0, padding_left_first_measure)
        xml_gen.add_sequence(score[1], notes_m21, bar_duration=bar_duration, measure_no=first_measure_num)
        xml_gen.remove_empty_measures(score)
    return score


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
    def __init__(self, file_list, min_sequence_length, max_sequence_length):
        """
        Makes a MusicXMLDataSet
        :param file_list: A list of MusicXML files to turn into a dataset
        :param min_sequence_length: The minimum sequence length
        :param max_sequence_length: The maximum sequence length
        """
        super(MusicXMLDataSet, self).__init__()
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.data, self.labels = self._load_data(file_list)
        
    def __len__(self) -> int:
        """
        Gets the number of entries in the dataset
        :return: The number of entries in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Gets the next item and label in the dataset
        :return: sample, label
        """
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, *label
    
    def _load_data(self, file_list) -> Tuple[list, list]:
        """
        Parses each MusicXML file and generates sequences and labels from it
        :param file_list: A list of MusicXML files to turn into a dataset
        """
        sequences = []
        labels = []
        for file in file_list:
            score = music21.corpus.parse(file)

            # Go through each staff in each score, and generate individual
            # sequences and labels for that staff
            for i in get_staff_indices(score):
                data = load_data(score[i])
                data = make_one_hot_features(data, False)
                for j in range(self.min_sequence_length, self.max_sequence_length + 1):
                    seq = make_n_gram_sequences(data, j+1)
                    lab = make_labels(seq)

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
        sequences, targets1, targets2, targets3, targets4 = zip(*batch)
        lengths = torch.tensor([seq.shape[0] for seq in sequences])
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        targets1 = torch.tensor(targets1)
        targets2 = torch.tensor(targets2)
        targets3 = torch.tensor(targets3)
        targets4 = torch.tensor(targets4)
        return sequences_padded, targets1, targets2, targets3, targets4, lengths
    
    def prepare_prediction(sequence, max_length):
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
    