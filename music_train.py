"""
File: music_train.py

Trains the music sequence generator
"""

import music_featurizer
import music_finder
import music_generator
import torch
import torch.nn as nn
import torch.optim as optim


def train(model, loss_fn_pitch_space, loss_fn_quarter_length, optimizer, training_x, training_y, batch_size, num_epochs, status_num=5, device="cpu"):
    """
    Trains the model
    :param model: The model to train
    :param loss_fn_pitch_space: The loss function for pitch space
    :param loss_fn_quarter_length: The loss function for quarter length
    :param optimizer: The optimizer
    :param training_x: The training data X
    :param training_y: The training data y
    :param batch_size: The size of each batch to run through LSTM
    :param num_epochs: The number of epochs for training
    :param status_num: How often (in epochs) to print an update message
    :param device: The device that is being used for the hidden matrices
    """
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_hat_pitch_space = []
        y_hat_quarter_length = []
        
        # Predict each sequence, in batches
        for i in range(0, training_x.shape[0], batch_size):
            size2 = min(batch_size, training_x.shape[0] - i)
            hidden = model.init_hidden(batch_size=size2, device=device)
            output, hidden = model(training_x[i:i+size2, :, :], hidden)
            y_hat_pitch_space.append(output[0])
            y_hat_quarter_length.append(output[1])

        # Compute loss and update weights
        y_hat_pitch_space = torch.vstack(y_hat_pitch_space)
        y_hat_quarter_length = torch.vstack(y_hat_quarter_length)
        loss_pitch_space = loss_fn_pitch_space(y_hat_pitch_space, training_y[0])
        loss_quarter_length = loss_fn_quarter_length(y_hat_quarter_length, training_y[1])
        total_loss = loss_pitch_space + loss_quarter_length
        total_loss.backward()
        optimizer.step()
        
        # Output status
        if epoch % status_num == status_num - 1:
            print(f"Epoch {epoch+1}, loss: {total_loss.item()}")


if __name__ == "__main__":
    PATH = "./data/train"
    TRAINING_SEQUENCE_LENGTH = 10
    OUTPUT_SIZE_PITCH_SPACE = len(music_featurizer._PS_ENCODING)
    OUTPUT_SIZE_QUARTER_LENGTH = len(music_featurizer._QUARTER_LENGTH_ENCODING)
    HIDDEN_SIZE = 1024
    NUM_LAYERS = 4
    LEARNING_RATE = 0.001
    TEMPO_DICT = {1: 100}

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}")
    
    # X, y_pitch_space, y_quarter_length = music_finder.prepare_directory(PATH, device)
    X, y_pitch_space, y_quarter_length = music_finder.prepare_m21_corpus('bach', device)
    print(X.shape)

    RETRAIN = True
    model = music_generator.LSTMMusic(music_featurizer._NUM_FEATURES, OUTPUT_SIZE_PITCH_SPACE, OUTPUT_SIZE_QUARTER_LENGTH, HIDDEN_SIZE, NUM_LAYERS).to(device)
    if RETRAIN:
        model.load_state_dict(torch.load("music_sequencer.pth"))
    loss_fn_pitch_space = nn.CrossEntropyLoss()
    loss_fn_quarter_length = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    NUM_EPOCHS = 400
    BATCH_SIZE = 100
    train(model, loss_fn_pitch_space, loss_fn_quarter_length, optimizer, X, (y_pitch_space, y_quarter_length), BATCH_SIZE, NUM_EPOCHS, status_num=5, device=device)
    
    # Save the model state
    torch.save(model.state_dict(), "music_sequencer.pth")
