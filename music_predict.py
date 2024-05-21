"""
File: music_predict.py
"""

import music21
import music_featurizer
import music_generator
import torch
import random


def predict_from_sequence(model, sequence):
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param sequence: The tokenized sequence of notes
    :return: The prediction as a MIDI note number, and the hidden states as a tuple
    """
    prediction = model(sequence, model.init_hidden())
    _, topk_indices = torch.topk(prediction[0], 12)
    topk_indices = topk_indices.reshape((topk_indices.shape[-1]))
    str_prediction = music_featurizer.retrieve_class_name(random.choice(topk_indices.tolist()))
    print(str_prediction, topk_indices.tolist())
    if str_prediction == "None":
        str_prediction = "-1"
    return float(str_prediction), prediction[1]


def predict_next_note(model, current_note, hidden):
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param current_note: The current note
    :param hidden: The hidden states
    :return: The prediction as a MIDI note number
    """
    prediction = model(current_note, hidden)
    _, topk_indices = torch.topk(prediction[0], 12)
    topk_indices = topk_indices.reshape((topk_indices.shape[-1]))
    str_prediction = music_featurizer.retrieve_class_name(random.choice(topk_indices.tolist()))
    print(str_prediction, topk_indices.tolist())
    if str_prediction == "None":
        str_prediction = "-1"
    return float(str_prediction), prediction[1]


if __name__ == "__main__":
    PATH = "data\\prompt_schutz.musicxml"
    TRAINING_SEQUENCE_LENGTH = 5
    NUM_FEATURES = 304
    OUTPUT_SIZE = 257
    HIDDEN_SIZE = 512
    
    random.seed()
    score = music21.converter.parse(PATH)
    STAFF_INDEX = 3

    data = music_featurizer.load_data(score[music_featurizer.get_staff_indices(score)[0]], {1: 100})[:-1]  # remove EOS
    X = music_featurizer.tokenize(data)
    
    # Load the model from file
    model = music_generator.LSTMMusic(NUM_FEATURES, OUTPUT_SIZE, HIDDEN_SIZE, 2)
    model.load_state_dict(torch.load("music_sequencer.pth"))
    
    # Predict the next note
    NOTES_TO_PREDICT = 5
    next_note, hidden = predict_from_sequence(model, X)    
    for i in range(NOTES_TO_PREDICT):
        if next_note < 0:
            letter_name = "None"
            pc = "None"
            accidental_alter = "None"
            next_note = "None"
        else:
            letter_name, pc, accidental_alter = music_featurizer.get_letter_name(next_note)
        current_note = {}
        current_note["BOS"] = False                                          # Beginning of part ("sentence")
        current_note["EOS"] = False                                          # End of part ("sentence")
        current_note["ps"] = next_note                                       # MIDI number (symbolic x257)
        current_note["letter_name"] = letter_name                            # letter name (C, D, E, ...) (symbolic x8)
        current_note["accidental"] = accidental_alter                        # accidental alteration (symbolic)
        current_note["pitch_class_id"] = pc                                  # pitch class number 0, 1, 2, ... 11 (symbolic x13)
        current_note["quarterLength"] = 1                                    # duration in quarter notes (symbolic)
        current_note["tempo"] = 100                                          # tempo (number)
        current_note["duration"] = 60 / 100 * current_note["quarterLength"]  # duration in seconds (number)
        data.append(current_note)
        next_note, hidden = predict_next_note(model, music_featurizer.tokenize([current_note]), hidden)

    score = music_featurizer.unload_data(data[1:])
    score.show()
