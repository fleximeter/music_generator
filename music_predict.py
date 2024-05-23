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
    s, l = music_featurizer.MusicXMLDataSet.prepare_prediction(sequence, 20)
    prediction, hidden = model(s, l, model.init_hidden())
    predicted_note = music_featurizer.retrieve_class_dictionary((prediction[0].argmax().item(), prediction[1].argmax().item()))
    return predicted_note, hidden


def predict_next_note(model, current_note, hidden):
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param current_note: The current note
    :param hidden: The hidden states
    :return: The prediction as a MIDI note number
    """
    s, l = music_featurizer.MusicXMLDataSet.prepare_prediction(current_note, 20)
    prediction, hidden = model(s, l, hidden)
    predicted_note = music_featurizer.retrieve_class_dictionary((prediction[0].argmax().item(), prediction[1].argmax().item()))
    return predicted_note, hidden


if __name__ == "__main__":
    PATH = "data\\prompt4.musicxml"
    OUTPUT_SIZE_PITCH_SPACE = len(music_featurizer._PS_ENCODING)
    OUTPUT_SIZE_QUARTER_LENGTH = len(music_featurizer._QUARTER_LENGTH_ENCODING)
    HIDDEN_SIZE = 1024
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
        X = music_featurizer.tokenize(data)
        next_note, hidden = predict_next_note(model, music_featurizer.tokenize([next_note]), hidden)

    score = music_featurizer.unload_data(data)
    #score.show()
