"""
File: music_featurizer.py

This module has featurization functions for preparing data to run through the model,
and for postprocessing generated data. It also has a dataset class for storing
sequences.
"""

import music21
import music_features
import torch
import torch.nn.functional as F
import math
import xml_gen
import numpy as np
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
        note["accidental"] = music_features.ACCIDENTAL_NAME_TO_ALTER_ENCODING[accidental_name]
        note["pitch_class_id"] = (PC_MAP[letter_name] + music_features.ACCIDENTAL_NAME_TO_ALTER_ENCODING[accidental_name]) % 12
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
        note["accidental_name"] = music_features.ACCIDENTAL_ALTER_TO_NAME_ENCODING[accidental_alter]
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
                        current_note["letter_accidental_name"] = f"{current_note["letter_name"]}|{current_note["accidental_name"]}"
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
                    current_note["letter_accidental_name"] = "None"
                    current_note["accidental"] = "None"                          # accidental alter value (symbolic)
                    current_note["pitch_class_id"] = "None"                      # pitch class number 0, 1, 2, ... 11 (symbolic x13)
                    current_note["key_signature"] = current_key                  # key signature
                    current_note["mode"] = current_mode                          # mode
                    current_note["quarterLength"] = item.duration.quarterLength  # duration in quarter notes (symbolic)
                    current_note["beat"] = item.beat                             # beat in quarter notes (symbolic)
                    current_note["time_signature"] = current_time_signature      # time signature (symbolic)
                    dataset.append(current_note)
                    current_note = {}

    # Fill melodic intervals
    last_ps = None
    for item in dataset:
        if item["ps"] != "None":
            if last_ps is not None:
                item["melodic_interval"] = (item["ps"] - last_ps) % 12
            else:
                item["melodic_interval"] = 0.0
            last_ps = item["ps"]
        else:
            item["melodic_interval"] = "None"
        
    return dataset


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
    for sequence in x:
        i = (
            (0, len(music_features.LETTER_ACCIDENTAL_ENCODING)), 
            (len(music_features.LETTER_ACCIDENTAL_ENCODING), len(music_features.LETTER_ACCIDENTAL_ENCODING) + len(music_features.OCTAVE_ENCODING)),
            (len(music_features.LETTER_ACCIDENTAL_ENCODING) + len(music_features.OCTAVE_ENCODING), len(music_features.LETTER_ACCIDENTAL_ENCODING) + len(music_features.OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING)),
            )
        letter_accidental = sequence[-1, i[0][0]:i[0][1]]
        octave = sequence[-1, i[1][0]:i[1][1]]
        quarter_length = sequence[-1, i[2][0]:i[2][1]]
        y.append((letter_accidental.argmax().item(), octave.argmax().item(), quarter_length.argmax().item()))
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
        letter_accidental_name_one_hot = F.one_hot(torch.tensor(music_features.LETTER_ACCIDENTAL_ENCODING[str(instance["letter_accidental_name"])]), len(music_features.LETTER_ACCIDENTAL_ENCODING)).float()
        octave_one_hot = F.one_hot(torch.tensor(music_features.OCTAVE_ENCODING[str(instance["octave"])]), len(music_features.OCTAVE_ENCODING)).float()
        if instance["quarterLength"] == "None":
            quarter_length_one_hot = F.one_hot(torch.tensor(music_features.QUARTER_LENGTH_ENCODING[str(instance["quarterLength"])]), len(music_features.QUARTER_LENGTH_ENCODING)).float()
        else:
            quarter_length_one_hot = F.one_hot(torch.tensor(music_features.QUARTER_LENGTH_ENCODING[str(Fraction(instance["quarterLength"]))]), len(music_features.QUARTER_LENGTH_ENCODING)).float()
        if instance["beat"] == "None":
            beat_one_hot = F.one_hot(torch.tensor(music_features.BEAT_ENCODING[str(instance["beat"])]), len(music_features.BEAT_ENCODING)).float()
        else:
            beat_one_hot = F.one_hot(torch.tensor(music_features.BEAT_ENCODING[str(Fraction(instance["beat"]))]), len(music_features.BEAT_ENCODING)).float()
        pitch_class_one_hot = F.one_hot(torch.tensor(music_features.PITCH_CLASS_ENCODING[str(instance["pitch_class_id"])]), len(music_features.PITCH_CLASS_ENCODING)).float()
        melodic_interval_one_hot = F.one_hot(torch.tensor(music_features.MELODIC_INTERVAL_ENCODING[str(instance["melodic_interval"])]), len(music_features.MELODIC_INTERVAL_ENCODING)).float()
        key_signature_one_hot = F.one_hot(torch.tensor(music_features.KEY_SIGNATURE_ENCODING[str(instance["key_signature"])]), len(music_features.KEY_SIGNATURE_ENCODING)).float()
        mode_one_hot = F.one_hot(torch.tensor(music_features.MODE_ENCODING[str(instance["mode"])]), len(music_features.MODE_ENCODING)).float()
        instances.append(torch.hstack((letter_accidental_name_one_hot, octave_one_hot, quarter_length_one_hot, beat_one_hot, 
                                       pitch_class_one_hot, melodic_interval_one_hot, key_signature_one_hot, mode_one_hot)))

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
    letter_name_accidental = music_features.REVERSE_LETTER_ACCIDENTAL_ENCODING[prediction[0]]
    letter, accidental = letter_name_accidental.split('|')
    note = {"letter_name_accidental": letter_name_accidental, "quarterLength": Fraction(music_features.REVERSE_QUARTER_LENGTH_ENCODING[prediction[2]])}
    note.update(convert_letter_accidental_octave_to_note(music_features.REVERSE_LETTER_NAME_ENCODING[letter], 
                                                         music_features.REVERSE_ACCIDENTAL_NAME_ENCODING[accidental], 
                                                         music_features.REVERSE_OCTAVE_ENCODING[prediction[1]]))
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


def update_note_based_on_previous(note, previous_notes):
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
