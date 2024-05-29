"""
File: featurizer.py

This module has featurization functions for preparing data to run through the model,
and for postprocessing generated data. It also has a dataset class for storing
sequences.
"""

import feature_definitions
import json
import torch
import torch.nn.functional as F
import math
from fractions import Fraction


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
        note["accidental"] = feature_definitions.ACCIDENTAL_NAME_TO_ALTER_ENCODING[accidental_name]
        note["pitch_class_id"] = (PC_MAP[letter_name] + feature_definitions.ACCIDENTAL_NAME_TO_ALTER_ENCODING[accidental_name]) % 12
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
        note["accidental_name"] = feature_definitions.ACCIDENTAL_ALTER_TO_NAME_ENCODING[accidental_alter]
    else:
        note["octave"] = "None"
        note["pitch_class_id"] = "None"
        note["ps"] = "None"
        note["letter_name"] = "None"
        note["accidental"] = "None"
        note["accidental_name"] = "None"

    return note


def load_json_corpus(json_corpus_file) -> list:
    """
    Loads a JSON corpus and prepares it for tokenization
    :param json_corpus_file: The file with the JSON corpus
    :return: The loaded corpus as a list (of lists of note dictionaries)
    """
    processed_score_list = []
    with open(json_corpus_file, "r") as output_json:
        processed_score_list = json.loads(output_json.read())
    for score in processed_score_list:
        for note in score:
            note["quarterLength"] = Fraction(note["quarterLength"])
            note["beat"] = Fraction(note["beat"])
    return processed_score_list


def make_labels(x) -> list:
    """
    Generates a label list for a list of sequences (2D tensors). The label is
    calculated for a particular index in dimension 1 of the 2D tensors.
    Dimension 1 is the batch length
    Dimension 2 is the total sequence length
    Dimension 3 is the features of individual notes in the sequence
    :param x: A list of sequences
    :return: A list of label tuples. Each label tuple has 3 labels (letter name + accidental name, octave, quarter length).
    """
    y = []
    i = (
        (0, len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING)), 
        (len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING))
        )
    for sequence in x:
        letter_accidental_octave = sequence[-1, i[0][0]:i[0][1]]
        quarter_length = sequence[-1, i[1][0]:i[1][1]]
        y.append((letter_accidental_octave.argmax().item(), quarter_length.argmax().item()))
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
        letter_accidental_octave_name_one_hot = F.one_hot(torch.tensor(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING[instance["letter_accidental_octave_name"]]), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING)).float()
        if instance["quarterLength"] == "None":
            quarter_length_one_hot = F.one_hot(torch.tensor(feature_definitions.QUARTER_LENGTH_ENCODING[str(instance["quarterLength"])]), len(feature_definitions.QUARTER_LENGTH_ENCODING)).float()
        else:
            quarter_length_one_hot = F.one_hot(torch.tensor(feature_definitions.QUARTER_LENGTH_ENCODING[str(Fraction(instance["quarterLength"]))]), len(feature_definitions.QUARTER_LENGTH_ENCODING)).float()
        if instance["beat"] == "None":
            beat_one_hot = F.one_hot(torch.tensor(feature_definitions.BEAT_ENCODING[str(instance["beat"])]), len(feature_definitions.BEAT_ENCODING)).float()
        else:
            beat_one_hot = F.one_hot(torch.tensor(feature_definitions.BEAT_ENCODING[str(Fraction(instance["beat"]))]), len(feature_definitions.BEAT_ENCODING)).float()
        pitch_class_one_hot = F.one_hot(torch.tensor(feature_definitions.PITCH_CLASS_ENCODING[str(instance["pitch_class_id"])]), len(feature_definitions.PITCH_CLASS_ENCODING)).float()
        melodic_interval_one_hot = F.one_hot(torch.tensor(feature_definitions.MELODIC_INTERVAL_ENCODING[str(instance["melodic_interval"])]), len(feature_definitions.MELODIC_INTERVAL_ENCODING)).float()
        key_signature_one_hot = F.one_hot(torch.tensor(feature_definitions.KEY_SIGNATURE_ENCODING[str(instance["key_signature"])]), len(feature_definitions.KEY_SIGNATURE_ENCODING)).float()
        mode_one_hot = F.one_hot(torch.tensor(feature_definitions.MODE_ENCODING[str(instance["mode"])]), len(feature_definitions.MODE_ENCODING)).float()
        time_signature_one_hot = F.one_hot(torch.tensor(feature_definitions.TIME_SIGNATURE_ENCODING[str(instance["time_signature"])]), len(feature_definitions.TIME_SIGNATURE_ENCODING)).float()
        instances.append(torch.hstack((letter_accidental_octave_name_one_hot, quarter_length_one_hot, beat_one_hot, 
                                       pitch_class_one_hot, melodic_interval_one_hot, key_signature_one_hot, mode_one_hot, time_signature_one_hot)))

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
    letter_accidental_octave_name = feature_definitions.REVERSE_LETTER_ACCIDENTAL_OCTAVE_ENCODING[prediction[0]]
    letter, accidental, octave = letter_accidental_octave_name.split('|')
    note = {"letter_accidental_octave_name": letter_accidental_octave_name, "letter_accidental_name": f"{letter}|{accidental}", "quarterLength": Fraction(feature_definitions.REVERSE_QUARTER_LENGTH_ENCODING[prediction[-1]])}
    # letter_accidental_name = music_features.REVERSE_LETTER_ACCIDENTAL_ENCODING[prediction[0]]
    # letter, accidental = letter_accidental_name.split('|')
    # octave = music_features.REVERSE_OCTAVE_ENCODING[prediction[1]]
    # note = {"letter_accidental_octave_name": f"{letter_accidental_name}|{octave}", "letter_accidental_name": letter_accidental_name, "quarterLength": Fraction(music_features.REVERSE_QUARTER_LENGTH_ENCODING[prediction[-1]])}
    note.update(convert_letter_accidental_octave_to_note(letter, accidental, octave))
    return note


def update_note_based_on_previous(note, previous_notes) -> None:
    """
    Updates a note based on its previous note. This is useful for adding in
    important details, like beat position, that cannot be determined immediately
    by the predictor.
    :param note: The note to update
    :param previous_notes: The previous notes in the sequence
    """
    time_signature_numerator = Fraction(previous_notes[-1]["time_signature"].split("/")[0])
    beat = Fraction(previous_notes[-1]["beat"]) + Fraction(previous_notes[-1]["quarterLength"])
    while beat > time_signature_numerator:
        beat -= time_signature_numerator
    note["key_signature"] = previous_notes[-1]["key_signature"] 
    note["mode"] = previous_notes[-1]["mode"]
    note["time_signature"] = previous_notes[-1]["time_signature"]
    note["beat"] = beat
    if note["ps"] == "None":
        note["melodic_interval"] = "None"
    else:
        note["melodic_interval"] = 0.0
        for i in range(len(previous_notes) - 1, -1, -1):
            if previous_notes[i]["ps"] != "None":
                note["melodic_interval"] = (note["ps"] - previous_notes[i]["ps"]) % 12
                break
