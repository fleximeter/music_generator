"""
File: music_train.py

Trains the music sequence generator
"""

import music21
import music_featurizer
import music_generator
import torch
import torch.nn as nn
import torch.optim as optim


def train(model, loss_fn, optimizer, training_x, training_y, batch_size, num_epochs, status_num=5, device="cpu"):
    """
    Trains the model
    :param model: The model to train
    :param loss_fn: The loss function
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
        y_hat = []
        
        # Predict each sequence, in batches
        for i in range(0, training_x.shape[0], batch_size):
            size2 = min(batch_size, training_x.shape[0] - i)
            hidden = model.init_hidden(batch_size=size2, device=device)
            output, hidden = model(training_x[i:i+size2, :, :], hidden)
            y_hat.append(output)

        # Compute loss and update weights
        y_hat = torch.vstack(y_hat)
        loss = loss_fn(y_hat, training_y)
        loss.backward()
        optimizer.step()
        
        # Output status
        if epoch % status_num == status_num - 1:
            print(f"Epoch {epoch+1}, loss: {loss.item()}")


if __name__ == "__main__":
    PATH = "data\\se_la_face_ay_pale.musicxml"
    TRAINING_SEQUENCE_LENGTH = 5
    NUM_FEATURES = 304
    OUTPUT_SIZE = 257
    HIDDEN_SIZE = 512
    LEARNING_RATE = 0.001

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}")
    
    score = music21.converter.parse(PATH)

    X = []
    y = []

    # Prepare the data for running through the model. We want sequences of length N for training.
    for i in music_featurizer.get_staff_indices(score):
        data = music_featurizer.load_data(score[i], {1: 100})
        data = music_featurizer.tokenize(data, False)
        data_x, data_y = music_featurizer.make_sequences(data, TRAINING_SEQUENCE_LENGTH, device=device)
        X.append(data_x)
        y.append(data_y)

    X = torch.vstack(X)
    y = torch.vstack(y)

    model = music_generator.LSTMMusic(NUM_FEATURES, OUTPUT_SIZE, HIDDEN_SIZE, 2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    NUM_EPOCHS = 100
    BATCH_SIZE = 50
    train(model, loss_fn, optimizer, X, y, BATCH_SIZE, NUM_EPOCHS, status_num=5, device=device)
    
    # Save the model state
    torch.save(model.state_dict(), "music_sequencer.pth")
