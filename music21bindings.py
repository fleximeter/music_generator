"""
File: music21bindings.py

This module has functions for performing music21 processing. It is separated from the other modules
so that the data can be prepared and later run on a system without music21.
"""

import music21
import numpy as np
import xml_gen


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
                        current_note["letter_accidental_name"] = f"{current_note['letter_name']}|{current_note['accidental_name']}"
                        current_note["letter_accidental_octave_name"] = f"{current_note['letter_name']}|{current_note['accidental_name']}|{current_note['octave']}"
                        current_note["pitch_class_id"] = float(item.pitch.pitchClass)          # pitch class number 0, 1, 2, ... 11 (symbolic x13)
                        current_note["key_signature"] = current_key                            # key signature
                        current_note["mode"] = current_mode                                    # mode

                        current_note["quarterLength"] = item.duration.quarterLength            # duration in quarter notes (symbolic)
                        current_note["beat"] = item.beat                                       # beat (symbolic)
                        current_note["time_signature"] = current_time_signature                # time signature (symbolic)

                        if item.tie is not None and item.tie.type in ["start", "continue"]:
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
                    current_note["letter_accidental_name"] = "None|None"
                    current_note["letter_accidental_octave_name"] = "None|None|None"
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
