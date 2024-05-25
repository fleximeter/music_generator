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


###################################################################################################################
# Feature dictionaries
###################################################################################################################

_LETTER_NAME_ENCODING = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6, "None": 7}
_ACCIDENTAL_ENCODING = {"-2.0": 0, "-1.0": 1, "0.0": 2, "1.0": 3, "2.0": 4, "None": 5}
_PITCH_SPACE_ENCODING = {"None": 0}
_OCTAVE_ENCODING = {"None": 0}
_PITCH_CLASS_ENCODING = {"None": 0}
_KEY_SIGNATURE_ENCODING = {"None": 0}
_MODE_ENCODING = {"None": 0, "major": 1, "minor": 2}
_BEAT_ENCODING = {"None": 0}
_QUARTER_LENGTH_ENCODING = {"None": 0}
_PITCH_SPACE_REVERSE_ENCODING = {0: "None"}
_OCTAVE_REVERSE_ENCODING = {0: "None"}
_PITCH_CLASS_REVERSE_ENCODING = {0: "None"}
_KEY_SIGNATURE_REVERSE_ENCODING = {0: "None"}
_MODE_REVERSE_ENCODING = {0: "None", 1: "major", 2: "minor"}
_BEAT_REVERSE_ENCODING = {0: "None"}
_QUARTER_LENGTH_REVERSE_ENCODING = {0: "None"}

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

j = 1

# Quarters
for i in range(1, 8):
    _QUARTER_LENGTH_ENCODING[f"{i}"] = j
    _QUARTER_LENGTH_REVERSE_ENCODING[j] = f"{i}"
    _BEAT_ENCODING[f"{i}"] = j
    _BEAT_REVERSE_ENCODING[j] = f"{i}"
    j += 1

# Eighths
for i in range(1, 16, 2):
    _QUARTER_LENGTH_ENCODING[f"{i}/2"] = j
    _QUARTER_LENGTH_REVERSE_ENCODING[j] = f"{i}/2"
    _BEAT_ENCODING[f"{i}/2"] = j
    _BEAT_REVERSE_ENCODING[j] = f"{i}/2"
    j += 1

# Triplet eighths
for i in range(1, 24):
    if i % 3 != 0:
        _QUARTER_LENGTH_ENCODING[f"{i}/3"] = j
        _QUARTER_LENGTH_REVERSE_ENCODING[j] = f"{i}/3"
        _BEAT_ENCODING[f"{i}/3"] = j
        _BEAT_REVERSE_ENCODING[j] = f"{i}/3"
        j += 1

# Sixteenths
for i in range(1, 32, 2):
    _QUARTER_LENGTH_ENCODING[f"{i}/4"] = j
    _QUARTER_LENGTH_REVERSE_ENCODING[j] = f"{i}/4"
    _BEAT_ENCODING[f"{i}/4"] = j
    _BEAT_REVERSE_ENCODING[j] = f"{i}/4"
    j += 1

# Triplet sixteenths
for i in range(1, 48, 2):
    if i % 3 != 0:
        _QUARTER_LENGTH_ENCODING[f"{i}/6"] = j
        _QUARTER_LENGTH_REVERSE_ENCODING[j] = f"{i}/6"
        _BEAT_ENCODING[f"{i}/6"] = j
        _BEAT_REVERSE_ENCODING[j] = f"{i}/6"
        j += 1

# 32nds
for i in range(1, 64, 2):
    _QUARTER_LENGTH_ENCODING[f"{i}/8"] = j
    _QUARTER_LENGTH_REVERSE_ENCODING[j] = f"{i}/8"
    _BEAT_ENCODING[f"{i}/8"] = j
    _BEAT_REVERSE_ENCODING[j] = f"{i}/8"
    j += 1


###################################################################################################################
# The total number of features and outputs for the model. This can change from time to time, and must be updated!
###################################################################################################################
_NUM_FEATURES = len(_PITCH_CLASS_ENCODING) + len(_OCTAVE_ENCODING) + len(_QUARTER_LENGTH_ENCODING) + len(_BEAT_ENCODING) + \
                len(_LETTER_NAME_ENCODING) + len(_ACCIDENTAL_ENCODING)  + len(_KEY_SIGNATURE_ENCODING)  + len(_MODE_ENCODING) 
_NUM_OUTPUTS = len(_PITCH_CLASS_ENCODING) + len(_OCTAVE_ENCODING) + len(_QUARTER_LENGTH_ENCODING)


def calculate_next_beat(note):
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


def convert_ps_to_note(ps: float):
    """
    Gets the letter name of a note and its accidental from a provided pitch space number
    :param ps: The pitch space number
    :return: The letter name and accidental alter value
    """
    PC_MAP = [('C', 0), ('C', -1), ('D', 0), ('D', -1), ('E', 0), ('F', 0), ('F', -1), ('G', 0), ('G', -1), ('A', 0), ('A', -1), ('B', 0)]
    microtone, semitone = math.modf(ps)
    pc = semitone % 12
    letter_name, accidental_alter = PC_MAP[int(pc)]
    accidental_alter -= microtone
    return letter_name, accidental_alter


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


def load_data(staff):
    """
    Loads a Music21 staff and featurizes it
    :param staff: The staff to load
    :return: The tokenized score
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

                        # accidental alteration (symbolic)
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
                    current_note["accidental"] = "None"                          # accidental name ("sharp", etc.) (symbolic)
                    current_note["pitch_class_id"] = "None"                      # pitch class number 0, 1, 2, ... 11 (symbolic x13)
                    current_note["key_signature"] = current_key                  # key signature
                    current_note["mode"] = current_mode                          # mode
                    current_note["quarterLength"] = item.duration.quarterLength  # duration in quarter notes (symbolic)
                    current_note["beat"] = item.beat                             # beat in quarter notes (symbolic)
                    current_note["time_signature"] = current_time_signature      # time signature (symbolic)
                    dataset.append(current_note)
                    current_note = {}

    return dataset


def make_labels(x):
    """
    Generates a label list for a list of sequences (2D tensors). The label is
    calculated for a particular index in dimension 1 of the 2D tensors.
    Dimension 1 is the batch length
    Dimension 2 is the total sequence length
    Dimension 3 is the features of individual notes in the sequence
    :param x: A list of sequences
    :return: A list of label tuples. Each label tuple has 3 labels (pc, octave, quarter length).
    """
    y = []
    for sequence in x:
        i = (
            (0, len(_PITCH_CLASS_ENCODING)), 
            (len(_PITCH_CLASS_ENCODING), len(_PITCH_CLASS_ENCODING) + len(_OCTAVE_ENCODING)),
            (len(_PITCH_CLASS_ENCODING) + len(_OCTAVE_ENCODING), len(_PITCH_CLASS_ENCODING) + len(_OCTAVE_ENCODING) + len(_QUARTER_LENGTH_ENCODING))
        )
        pc = sequence[-1, i[0][0]:i[0][1]]
        octave = sequence[-1, i[1][0]:i[1][1]]
        quarter_length = sequence[-1, i[2][0]:i[2][1]]
        y.append((pc.argmax().item(), octave.argmax().item(), quarter_length.argmax().item()))
    return y


def make_n_gram_sequences(tokenized_dataset, n):
    """
    Makes N-gram sequences from a tokenized dataset
    :param tokenized_dataset: The tokenized dataset
    :param n: The length of the n-grams
    :return: X, (y1, y2) (a 3D tensor and a tuple of 2D tensors).
    X dimension 1 is the N-gram
    X dimension 2 is each entry in the N-gram
    X dimension 3 is the features of the entry
    """
    x = []
    for j in range(n, tokenized_dataset.shape[0] - 1):
        y = []
        for k in range(j-n, j):
            y.append(tokenized_dataset[k, :])
        x.append(torch.vstack(y))
    return x


def make_one_hot_features(dataset, batched=True):
    """
    Turns a dataset into a list of one-hot-featured instances in preparation for 
    running it through a model. You can use this for making predictions if you want.
    :param dataset: The dataset to make one-hot
    :param batched: Whether or not the data is expected to be in batched format (3D) 
    or unbatched format (2D). If you will be piping this data into the make_sequences 
    function, it should not be batched. In all other cases, it should be batched.
    :return: The one-hot data
    """
    instances = []
    for instance in dataset:
        # One-hots
        pitch_space_one_hot = F.one_hot(torch.tensor(_PITCH_CLASS_ENCODING[str(instance["pitch_class_id"])]), len(_PITCH_CLASS_ENCODING)).float()
        octave_one_hot = F.one_hot(torch.tensor(_OCTAVE_ENCODING[str(instance["octave"])]), len(_OCTAVE_ENCODING)).float()
        if instance["quarterLength"] == "None":
            quarter_length_one_hot = F.one_hot(torch.tensor(_QUARTER_LENGTH_ENCODING[str(instance["quarterLength"])]), len(_QUARTER_LENGTH_ENCODING)).float()
        else:
            quarter_length_one_hot = F.one_hot(torch.tensor(_QUARTER_LENGTH_ENCODING[str(Fraction(instance["quarterLength"]))]), len(_QUARTER_LENGTH_ENCODING)).float()
        if instance["beat"] == "None":
            beat_one_hot = F.one_hot(torch.tensor(_BEAT_ENCODING[str(instance["beat"])]), len(_BEAT_ENCODING)).float()
        else:
            beat_one_hot = F.one_hot(torch.tensor(_BEAT_ENCODING[str(Fraction(instance["beat"]))]), len(_BEAT_ENCODING)).float()
        letter_one_hot = F.one_hot(torch.tensor(_LETTER_NAME_ENCODING[str(instance["letter_name"])]), len(_LETTER_NAME_ENCODING)).float()
        accidental_one_hot = F.one_hot(torch.tensor(_ACCIDENTAL_ENCODING[str(instance["accidental"])]), len(_ACCIDENTAL_ENCODING)).float()
        key_signature_one_hot = F.one_hot(torch.tensor(_KEY_SIGNATURE_ENCODING[str(instance["key_signature"])]), len(_KEY_SIGNATURE_ENCODING)).float()
        mode_one_hot = F.one_hot(torch.tensor(_MODE_ENCODING[str(instance["mode"])]), len(_MODE_ENCODING)).float()
        instances.append(torch.hstack((pitch_space_one_hot, octave_one_hot, quarter_length_one_hot, beat_one_hot, 
                                       letter_one_hot, accidental_one_hot, key_signature_one_hot, mode_one_hot)))

    instances = torch.vstack(instances)
    if batched:
        instances = torch.reshape(instances, (1,) + instances.shape)
    return instances


def retrieve_class_dictionary(prediction):
    """
    Retrives a predicted class's information based on its id
    :param prediction: The prediction tuple
    :return: The prediction dictionary
    """
    pc = _PITCH_CLASS_REVERSE_ENCODING[prediction[0]]
    octave = _OCTAVE_REVERSE_ENCODING[prediction[1]]
    if pc != "None" and octave != "None":
        octave = int(octave)
        pc = float(pc)
        ps = pc + (octave + 1) * 12
        letter_name, accidental_alter = convert_ps_to_note(ps)
    else:
        letter_name = "None"
        pc = "None"
        ps = "None"
        accidental_alter = "None"

    return {
        "ps": ps,
        "octave": octave,
        "quarterLength": Fraction(_QUARTER_LENGTH_REVERSE_ENCODING[prediction[2]]),
        "letter_name": letter_name,
        "accidental": accidental_alter,
        "pitch_class_id": pc,
    }


def unload_data(dataset):
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
        if item["ps"] == "None":
            notes.append(-np.inf)
        else:
            notes.append(float(item["ps"]))
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
        xml_gen.fix_last_measure(score)
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
        
    def __len__(self):
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
        return sample, label[0], label[1], label[2]
    
    def _load_data(self, file_list):
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
        sequences, targets1, targets2, targets3 = zip(*batch)
        lengths = torch.tensor([seq.shape[0] for seq in sequences])
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        targets1 = torch.tensor(targets1)
        targets2 = torch.tensor(targets2)
        targets3 = torch.tensor(targets3)
        return sequences_padded, targets1, targets2, targets3, lengths
    
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
    