"""
File: music_predict.py

This module makes predictions based on an existing model that was
saved to file.
"""

import music21
import music_featurizer
import music_generator
import torch


def predict_from_sequence(model, sequence):
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param sequence: The tokenized sequence of notes
    :return: The prediction as a MIDI note number, and the hidden states as a tuple
    """
    prediction, hidden = model(sequence, model.init_hidden())
    prediction_pitch_space = torch.reshape(prediction[0], (prediction[0].size()))
    prediction_quarter_length = torch.reshape(prediction[1], (prediction[1].size()))
    prediction_idx_pitch_space = prediction_pitch_space.argmax().item()
    prediction_idx_quarter_length = prediction_quarter_length.argmax().item()
    predicted_note = music_featurizer.retrieve_class_dictionary((prediction_idx_pitch_space, prediction_idx_quarter_length))
    return predicted_note, hidden


def predict_next_note(model, current_note, hidden):
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param current_note: The current note
    :param hidden: The hidden states
    :return: The prediction as a MIDI note number
    """
    prediction, hidden = model(current_note, hidden)
    prediction_pitch_space = torch.reshape(prediction[0], (prediction[0].size()))
    prediction_quarter_length = torch.reshape(prediction[1], (prediction[1].size()))
    prediction_idx_pitch_space = prediction_pitch_space.argmax().item()
    prediction_idx_quarter_length = prediction_quarter_length.argmax().item()
    predicted_note = music_featurizer.retrieve_class_dictionary((prediction_idx_pitch_space, prediction_idx_quarter_length))
    return predicted_note, hidden


if __name__ == "__main__":
    PATH = "data\\prompt3.musicxml"
    OUTPUT_SIZE_PITCH_SPACE = len(music_featurizer._PS_ENCODING)
    OUTPUT_SIZE_QUARTER_LENGTH = len(music_featurizer._QUARTER_LENGTH_ENCODING)
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4
    TEMPO_DICT = {1: 100}
    
    # Predict only for the top staff
    score = music21.converter.parse(PATH)
    data = music_featurizer.load_data(score[music_featurizer.get_staff_indices(score)[0]], TEMPO_DICT)
    X = music_featurizer.tokenize(data)
    
    # Load the model from file
    model = music_generator.LSTMMusic(music_featurizer._NUM_FEATURES, OUTPUT_SIZE_PITCH_SPACE, OUTPUT_SIZE_QUARTER_LENGTH, HIDDEN_SIZE, NUM_LAYERS)
    model.load_state_dict(torch.load("music_sequencer_1.pth"))
    
    # Predict the next N notes
    NOTES_TO_PREDICT = 10
    next_note, hidden = predict_from_sequence(model, X)
    for i in range(NOTES_TO_PREDICT):
        data.append(next_note)
        next_note, hidden = predict_next_note(model, music_featurizer.tokenize([next_note]), hidden)

    score = music_featurizer.unload_data(data)
    score.show()
