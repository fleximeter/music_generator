"""
File: music_predict.py

This module makes predictions based on an existing model that was
saved to file.
"""

import json
import music21
import music_featurizer
import music_generator
import torch


def predict_from_sequence(model, sequence, training_sequence_max_length):
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param sequence: The tokenized sequence of notes
    :param training_sequence_max_length: The maximum sequence length the model was trained on
    :return: The prediction as a MIDI note number, and the hidden states as a tuple
    """
    s, l = music_featurizer.MusicXMLDataSet.prepare_prediction(sequence, training_sequence_max_length)
    prediction, hidden = model(s, l, model.init_hidden())
    predicted_note = music_featurizer.retrieve_class_dictionary((prediction[0].argmax().item(), prediction[1].argmax().item(), prediction[2].argmax().item()))
    return predicted_note, hidden


def predict_next_note(model, current_note, hidden, training_sequence_max_length):
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param current_note: The current note
    :param hidden: The hidden states
    :param training_sequence_max_length: The maximum sequence length the model was trained on
    :return: The prediction as a MIDI note number
    """
    s, l = music_featurizer.MusicXMLDataSet.prepare_prediction(current_note, training_sequence_max_length)
    prediction, hidden = model(s, l, hidden)
    predicted_note = music_featurizer.retrieve_class_dictionary((prediction[0].argmax().item(), prediction[1].argmax().item(), prediction[2].argmax().item()))
    return predicted_note, hidden


if __name__ == "__main__":
    PATH = "./data/prompt3.musicxml"
    FILE_NAME = "./data/model.json"
    TEMPO_DICT = {1: 100}
    
    # Load the model information
    model_data = None
    with open(FILE_NAME, "r") as model_json_file:
        model_data = json.loads(model_json_file.read())
    
    # Predict only for the top staff
    score = music21.converter.parse(PATH)
    data = music_featurizer.load_data(score[music_featurizer.get_staff_indices(score)[0]], TEMPO_DICT)
    tokenized_data = music_featurizer.tokenize(data)
    
    # Load the model from file
    model = music_generator.LSTMMusic(model_data["num_features"], model_data["output_size_pc"], model_data["output_size_octave"], 
                                      model_data["output_size_quarter_length"], model_data["hidden_size"], 
                                      model_data["num_layers"])
    model.load_state_dict(torch.load(model_data["state_dict"]))
    
    # Predict the next N notes
    NOTES_TO_PREDICT = 10
    next_note, hidden = predict_from_sequence(model, tokenized_data, model_data["training_sequence_max_length"])
    for i in range(NOTES_TO_PREDICT):
        data.append(next_note)
        next_note, hidden = predict_next_note(model, music_featurizer.tokenize([next_note]), hidden, model_data["training_sequence_max_length"])

    score = music_featurizer.unload_data(data)
    score.show()
