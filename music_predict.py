"""
File: music_predict.py
"""

import music21
import music_featurizer
import music_generator
import torch

if __name__ == "__main__":
    PATH = "data\\ein_kind_ist_uns_geboren.xml"
    TRAINING_SEQUENCE_LENGTH = 5
    NUM_FEATURES = 304
    OUTPUT_SIZE = 257
    HIDDEN_SIZE = 512
    
    score = music21.converter.parse(PATH)

    X = []
    y = []

    # Prepare the data for running through the model. We want sequences of length N for training.
    for i in range(1, 7):
        data = music_featurizer.load_data(score[i], {1: 100})
        data = music_featurizer.tokenize(data)
        data_x, data_y = music_featurizer.make_sequences(data, TRAINING_SEQUENCE_LENGTH)
        X.append(data_x)
        y.append(data_y)

    X = torch.vstack(X)
    y = torch.vstack(y)

    # Load the model from file
    model = music_generator.LSTMMusic(NUM_FEATURES, OUTPUT_SIZE, HIDDEN_SIZE, 2)
    model.load_state_dict(torch.load("music_sequencer.pth"))
    
