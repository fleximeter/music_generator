"""
File: music_predict.py

This module makes predictions based on an existing model that was saved to file.
You will need to provide the model metadata file name so that it can load important
information about the model, such as the number of layers and the hidden size.
"""

import dataset
import json
import music21
import music_featurizer
import music_generator
import torch
from typing import Tuple


def predict_from_sequence(model, sequence, training_sequence_max_length) -> Tuple[dict, torch.Tensor]:
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param sequence: The tokenized sequence of notes
    :param training_sequence_max_length: The maximum sequence length the model was trained on.
    This is necessary because the DataLoader will pad sequences that are shorter than the maximum
    length, and the model might not behave as predicted if we don't pad sequences that we use
    as prompts.
    :return: The prediction as a note dictionary, and the hidden states as a tuple
    """
    s, l = dataset.MusicXMLDataSet.prepare_prediction(sequence, training_sequence_max_length)
    prediction, hidden = model(s, l, model.init_hidden())
    predicted_note = music_featurizer.retrieve_class_dictionary((prediction[0].argmax().item(), prediction[1].argmax().item(), 
                                                                 prediction[-1].argmax().item()))
    return predicted_note, hidden


def predict_next_note(model, current_note, hidden, training_sequence_max_length) -> Tuple[dict, torch.Tensor]:
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param current_note: The current note
    :param hidden: The hidden states
    :param training_sequence_max_length: The maximum sequence length the model was trained on
    This is necessary because the DataLoader will pad sequences that are shorter than the maximum
    length, and the model might not behave as predicted if we don't pad sequences that we use
    as prompts.
    :return: The prediction as a note dictionary, and the hidden states as a tuple
    """
    s, l = dataset.MusicXMLDataSet.prepare_prediction(current_note, training_sequence_max_length)
    prediction, hidden = model(s, l, hidden)
    predicted_note = music_featurizer.retrieve_class_dictionary((prediction[0].argmax().item(), prediction[1].argmax().item(), 
                                                                 prediction[-1].argmax().item()))
    return predicted_note, hidden


if __name__ == "__main__":
    #######################################################################################
    # YOU WILL NEED TO EDIT THIS MANUALLY
    #######################################################################################

    MUSICXML_PROMPT_FILE = "./data/prompt7.musicxml"  # Only the top staff will be considered
    MODEL_METADATA_FILE = "./data/model13.json"
    NOTES_TO_PREDICT = 25

    #######################################################################################
    # YOU PROBABLY DON'T NEED TO EDIT ANYTHING BELOW HERE
    #######################################################################################
    
    # Load the model information
    model_metadata = None
    abort = False

    try:
        with open(MODEL_METADATA_FILE, "r") as model_json_file:
            model_metadata = json.loads(model_json_file.read())
    except FileNotFoundError as e:
        abort = True
        print("ERROR: Could not open the model metadata file. Aborting.")
    
    try:
        score = music21.converter.parse(MUSICXML_PROMPT_FILE)
    except Exception as e:
        abort = True
        print("ERROR: Could not read the Music XML prompt file. Aborting.")

    if not abort:
        # Predict only for the top staff
        data = music_featurizer.load_data(score[music_featurizer.get_staff_indices(score)[0]])
        tokenized_data = music_featurizer.make_one_hot_features(data)
        
        # Load the model state dictionary from file
        model = music_generator.LSTMMusic(model_metadata["num_features"], model_metadata["output_sizes"], 
                                          model_metadata["hidden_size"], model_metadata["num_layers"])
        model.load_state_dict(torch.load(model_metadata["state_dict"]))
        
        # Predict the next N notes
        next_note, hidden = predict_from_sequence(model, tokenized_data, model_metadata["training_sequence_max_length"])
        for i in range(NOTES_TO_PREDICT):
            # get the note time signature and beat based on the previous note
            music_featurizer.update_note_based_on_previous(next_note, data)
            data.append(next_note)
            next_note, hidden = predict_next_note(model, music_featurizer.make_one_hot_features([next_note]), hidden, model_metadata["training_sequence_max_length"])

        # Turn the data into a score
        score = music_featurizer.unload_data(data)
        score.show()
        # destination_split = os.path.split(MUSICXML_PROMPT_FILE)
        # destination_file = "predicted_" + destination_split[-1]
        # xml_gen.export_to_xml(score, os.path.join(*destination_split[:-1], destination_file))
