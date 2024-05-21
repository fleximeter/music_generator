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
    prediction, hidden = model(sequence, model.init_hidden())
    # _, topk_indices = torch.topk(prediction[0], 12)
    # topk_indices = topk_indices.reshape((topk_indices.shape[-1]))
    prediction_idx_pitch_space = prediction[0].argmax().item()
    prediction_idx_quarter_length = prediction[1].argmax().item()
    str_prediction = music_featurizer.retrieve_class_dictionary((prediction_idx_pitch_space, prediction_idx_quarter_length))
    # print(str_prediction, topk_indices.tolist())
    # if str_prediction == "None":
    #     str_prediction = "-1"
    return str_prediction, hidden


def predict_next_note(model, current_note, hidden):
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param current_note: The current note
    :param hidden: The hidden states
    :return: The prediction as a MIDI note number
    """
    prediction, hidden = model(current_note, hidden)
    # _, topk_indices = torch.topk(prediction[0], 12)
    # topk_indices = topk_indices.reshape((topk_indices.shape[-1]))
    prediction_idx_pitch_space = prediction[0].argmax().item()
    prediction_idx_quarter_length = prediction[1].argmax().item()
    str_prediction = music_featurizer.retrieve_class_dictionary((prediction_idx_pitch_space, prediction_idx_quarter_length))
    return str_prediction, hidden


if __name__ == "__main__":
    PATH = "data\\prompt1.musicxml"
    OUTPUT_SIZE_PITCH_SPACE = len(music_featurizer._PS_ENCODING)
    OUTPUT_SIZE_QUARTER_LENGTH = len(music_featurizer._QUARTER_LENGTH_ENCODING)
    HIDDEN_SIZE = 1024
    NUM_LAYERS = 4
    TEMPO_DICT = {1: 100}
    
    random.seed()
    score = music21.converter.parse(PATH)
    STAFF_INDEX = 3

    data = music_featurizer.load_data(score[music_featurizer.get_staff_indices(score)[0]], TEMPO_DICT)[:-1]  # remove EOS
    X = music_featurizer.tokenize(data)
    
    # Load the model from file
    model = music_generator.LSTMMusic(music_featurizer._NUM_FEATURES, OUTPUT_SIZE_PITCH_SPACE, OUTPUT_SIZE_QUARTER_LENGTH, HIDDEN_SIZE, NUM_LAYERS)
    model.load_state_dict(torch.load("music_sequencer.pth"))
    
    # Predict the next note
    NOTES_TO_PREDICT = 5
    next_note, hidden = predict_from_sequence(model, X)    
    for i in range(NOTES_TO_PREDICT):
        next_note["tempo"] = TEMPO_DICT[1]                                            # tempo (number)
        next_note["duration"] = 60 / next_note["tempo"] * next_note["quarterLength"]  # duration in seconds (number)
        data.append(next_note)
        next_note, hidden = predict_next_note(model, music_featurizer.tokenize([next_note]), hidden)

    score = music_featurizer.unload_data(data[1:])
    score.show()
