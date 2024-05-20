"""
File: music_featurizer.py

Featurizes a music21 staff for running through LSTM
"""

import music21
import torch

_LETTER_NAME_ENCODING = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6, "None": 7}
_ACCIDENTAL_ENCODING = {"-2.0": 0, "-1.5": 1, "-1.0": 2, "-0.5": 3, "0.0": 4, "0.5": 5, "1.0": 6, "1.5": 7, "2.0": 8, "None": 9}
_PS_ENCODING = {"None": 256}
_PITCH_CLASS_ENCODING = {"None": 24}
_QUARTER_LENGTH_ENCODING = {"None": 0}

for i in range(256):
    _PS_ENCODING[str(i/2)] = i 

for i in range(24):
    _PITCH_CLASS_ENCODING[str(i/2)] = i 

j = 1

# Quarters
for i in range(1, 8):
    _QUARTER_LENGTH_ENCODING[f"{i}"] = j
    j += 1

# Eighths
for i in range(1, 16, 2):
    _QUARTER_LENGTH_ENCODING[f"{i}/2"] = j
    j += 1

# Triplet eighths
for i in range(1, 24):
    if i % 3 != 0:
        _QUARTER_LENGTH_ENCODING[f"{i}/3"] = j
        j += 1

# Sixteenths
for i in range(1, 32, 2):
    _QUARTER_LENGTH_ENCODING[f"{i}/4"] = j
    j += 1

# Triplet sixteenths
for i in range(1, 48, 2):
    if i % 3 != 0:
        _QUARTER_LENGTH_ENCODING[f"{i}/6"] = j
        j += 1

# 32nds
for i in range(1, 64, 2):
    _QUARTER_LENGTH_ENCODING[f"{i}/8"] = j
    j += 1


def load_data(staff, tempo_map):
    """
    Loads a Music21 staff and featurizes it
    :param staff: The staff to load
    :param tempo_map: A tempo dictionary. Keys are measure numbers, and values are quarter note tempi.
    :return: The tokenized score
    """
    dataset = []

    tempo = 60

    # Add BOS
    dataset.append({"BOS": True, "EOS": False, "ps": "None", "letter_name": "None", "accidental": "None", "pitch_class_id": "None", "quarterLength": "None", "tempo": "None", "duration": "None"})

    tie_status = False
    current_note = {}

    # We assume there are not multiple voices on this staff, and there are no chords - it's just a line
    for measure in staff:
        if type(measure) == music21.stream.Measure:
            for item in measure:
                if type(item) == music21.note.Note:
                    if item.measureNumber in tempo_map:
                        tempo = tempo_map[item.measureNumber]
                    if not tie_status:
                        current_note["BOS"] = False                                            # Beginning of part ("sentence")
                        current_note["EOS"] = False                                            # End of part ("sentence")
                        current_note["ps"] = item.pitch.ps                                     # MIDI number (symbolic x257)
                        current_note["letter_name"] = item.pitch.step                          # letter name (C, D, E, ...) (symbolic x8)

                        # accidental alteration (symbolic)
                        current_note["accidental"] = item.pitch.accidental.alter if item.pitch.accidental is not None else 0.0
                        current_note["pitch_class_id"] = float(item.pitch.pitchClass)          # pitch class number 0, 1, 2, ... 11 (symbolic x13)
                        current_note["quarterLength"] = item.duration.quarterLength            # duration in quarter notes (symbolic)
                        current_note["tempo"] = tempo                                          # tempo (number)
                        current_note["duration"] = 60 / tempo * current_note["quarterLength"]  # duration in seconds (number)

                        if item.tie is not None and item.tie.type in ["continue", "stop"]:
                            tie_status = True
                        else:
                            dataset.append(current_note)
                            current_note = {}
                        
                    else:
                        current_note["quarterLength"] += item.duration.quarterLength
                        current_note["duration"] += 60 / tempo * item.duration.quarterLength
                        if item.tie is None or item.tie.type != "continue":
                            tie_status = False
                            dataset.append(current_note)
                            current_note = {}

                if type(item) == music21.note.Rest:
                    if item.measureNumber in tempo_map:
                        tempo = tempo_map[item.measureNumber]
                    tie_status = False
                    current_note["BOS"] = False                                            # Beginning of part ("sentence")
                    current_note["EOS"] = False                                            # End of part ("sentence")
                    current_note["ps"] = "None"                                            # MIDI number (symbolic x257)
                    current_note["letter_name"] = "None"                                   # letter name (C, D, E, ...) (symbolic x8)
                    current_note["accidental"] = "None"                                    # accidental name ("sharp", etc.) (symbolic)
                    current_note["pitch_class_id"] = "None"                                # pitch class number 0, 1, 2, ... 11 (symbolic x13)
                    current_note["quarterLength"] = item.duration.quarterLength            # duration in quarter notes (symbolic)
                    current_note["tempo"] = tempo                                          # tempo (number)
                    current_note["duration"] = 60 / tempo * current_note["quarterLength"]  # duration in seconds (number)
                    dataset.append(current_note)
                    current_note = {}
            
    # Add EOS
    dataset.append({"BOS": False, "EOS": True, "ps": "None", "letter_name": "None", "accidental": "None", "pitch_class_id": "None", "quarterLength": "None", "tempo": "None", "duration": "None"})

    return dataset


def tokenize(dataset, batched=True):
    """
    Tokenize a dataset. You can use this for making predictions if you want.
    :param dataset: The dataset to tokenize
    :param batched: Whether or not the data is expected to be in batched 
    format (3D) or unbatched format (2D). If you will be piping this data 
    into the make_sequences function, it should not be batched. In all 
    other cases, it should be batched.
    :return: The tokenized data
    """
    entries = []
    for entry in dataset:
        bos = torch.Tensor([int(entry["BOS"]), int(entry["EOS"])])
        ps = torch.zeros((len(_PS_ENCODING)))
        ps[_PS_ENCODING[str(entry["ps"])]] = 1
        letter = torch.zeros((len(_LETTER_NAME_ENCODING)))
        letter[_LETTER_NAME_ENCODING[entry["letter_name"]]] = 1
        accidental = torch.zeros((len(_ACCIDENTAL_ENCODING)))
        accidental[_ACCIDENTAL_ENCODING[str(entry["accidental"])]] = 1
        pc = torch.zeros((len(_PITCH_CLASS_ENCODING)))
        pc[_PITCH_CLASS_ENCODING[str(entry["pitch_class_id"])]] = 1
        if entry["tempo"] == "None" or entry["duration"] == "None":
            tempo = torch.zeros((2))
        else:
            tempo = torch.Tensor([entry["tempo"], entry["duration"]])
        entries.append(torch.hstack((bos, ps, letter, accidental, pc, tempo)))

    entries = torch.vstack(entries)
    if batched:
        entries = torch.reshape(entries, (1, entries.shape[0], entries.shape[1]))
    return entries


def retrieve_class_name(class_id):
    """
    Retrives a predicted class's name based on its id
    :param class_id: The class id
    :return: The class name
    """
    for item, val in _PS_ENCODING.items():
        if class_id == val:
            return item
    return "None"


def make_sequences(tokenized_dataset, n, device="cpu"):
    """
    Makes N-gram sequences from a tokenized dataset
    :param tokenized_dataset: The tokenized dataset
    :param n: The length of the n-grams
    :param device: The device on which to initialize the tensors (cpu, cuda, mps)
    :return: X, y (a 3D tensor and a 2D tensor).
    X dimension 1 is the N-gram
    X dimension 2 is each entry in the N-gram
    X dimension 3 is the features of the entry

    y dimension 1 is the y for the corresponding N-gram
    y dimension 2 is the features of the y
    """
    x = []
    y = []
    num_features = tokenized_dataset.shape[-1]

    for i in range(n):
        new_x = []
        if n-i-1 > 0:
            new_x.append(torch.zeros((n-i-1, num_features)))
        new_x.append(tokenized_dataset[:i+1, :])
        x.append(torch.vstack(new_x))
        y.append(tokenized_dataset[i+1, 2:2+257])
    
    for j in range(n, tokenized_dataset.shape[0] - 1):
        x.append(tokenized_dataset[j-n:j, :])
        y.append(tokenized_dataset[j+1, 2:2+257])
    
    x = torch.stack(x, dim=0)
    y = torch.vstack(y)
    
    return x.to(device), y.to(device)
    