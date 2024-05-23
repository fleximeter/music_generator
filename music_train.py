"""
File: music_train.py

This module trains the music sequence generator.
"""

import datetime
import music_featurizer
import music_finder
import music_generator
import torch
import torch.nn as nn
import torch.optim as optim


def train_sequences(model, loss_fn_pitch_space, loss_fn_quarter_length, optimizer, training_data_x, min_length, max_length, batch_size, num_epochs, status_num=5, save_every=100, file="music_sequencer_1.pth", device="cpu"):
    """
    Trains the model. This training function expects a batch of data X, which will be turned into 
    sequences of different lengths from min_length to max_length. The labels y are calculated as
    part of the sequencing process. Each epoch, all sequences of all lengths are processed once.
    :param model: The model to train
    :param loss_fn_pitch_space: The loss function for pitch space
    :param loss_fn_quarter_length: The loss function for quarter length
    :param optimizer: The optimizer
    :param training_data_x: The training data X
    :param min_length: The minimum sequence length
    :param max_length: The maximum sequence length
    :param batch_size: The size of each batch to run through LSTM
    :param num_epochs: The number of epochs for training
    :param status_num: How often (in epochs) to print an update message
    :param save_every: Save every N times to disk
    :param file: The file name (for saving to disk)
    :param device: The device that is being used for the hidden matrices
    """

    # Generate sequences of different lengths. For each length, a sequence dictionary is stored.
    # This dictionary has the X and y values in it.
    sequences = []
    for i in range(min_length, max_length + 1):
        data_x, data_y = music_featurizer.make_n_gram_sequences(training_data_x, i, device=device)
        sequences.append({"sequence_length": i, "X": data_x, "y_ps": data_y[0], "y_ql": data_y[1]})

    # Train for N epochs
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Holds the predictions
        y_hat_pitch_space = []
        y_hat_quarter_length = []

        # Holds the corresponding correct labels
        y_pitch_space = []
        y_quarter_length = []
        
        # Predict sequences of different lengths. Each time the outer loop runs,
        # the sequence length will be different.
        for i, sequences_n in enumerate(sequences):
            # Predict each sequence of length n, in batches
            for j in range(0, sequences_n["X"].shape[0], batch_size):
                size2 = min(batch_size, sequences_n["X"].shape[0] - j)
                hidden = model.init_hidden(batch_size=size2, device=device)
                output, hidden = model(sequences_n["X"][j:j+size2, :, :], hidden)
                
                # Record the proper labels
                y_pitch_space.append(sequences_n["y_ps"][j:j+size2])
                y_quarter_length.append(sequences_n["y_ql"][j:j+size2])
                
                # Record the predicted labels
                y_hat_pitch_space.append(output[0])
                y_hat_quarter_length.append(output[1])


        # Combine the predicted labels and the actual labels, in preparation for loss calculation
        y_pitch_space = torch.hstack(y_pitch_space)
        y_quarter_length = torch.hstack(y_quarter_length)
        y_hat_pitch_space = torch.vstack(y_hat_pitch_space)
        y_hat_quarter_length = torch.vstack(y_hat_quarter_length)

        # Compute loss and update weights
        loss_pitch_space = loss_fn_pitch_space(y_hat_pitch_space, y_pitch_space)
        loss_quarter_length = loss_fn_quarter_length(y_hat_quarter_length, y_quarter_length)
        total_loss = loss_pitch_space + loss_quarter_length
        total_loss.backward()
        optimizer.step()
        
        # Output status
        if epoch % status_num == status_num - 1:
            print(f"Epoch {epoch+1:<4} | loss: {round(total_loss.item(), 8):<12} | time: {datetime.datetime.now()}")

        if epoch % save_every == save_every - 1:
            torch.save(model.state_dict(), file)



if __name__ == "__main__":
    PATH = "./data/train"
    TRAINING_SEQUENCE_MAX_LENGTH = 16
    TRAINING_SEQUENCE_MIN_LENGTH = 2
    OUTPUT_SIZE_PITCH_SPACE = len(music_featurizer._PS_ENCODING)
    OUTPUT_SIZE_QUARTER_LENGTH = len(music_featurizer._QUARTER_LENGTH_ENCODING)
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4
    LEARNING_RATE = 0.001

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}")
    
    # X = music_finder.prepare_directory(PATH, device)
    X = music_finder.prepare_m21_corpus('bach', device)
    
    # Whether or not to continue training the same model
    RETRAIN = True
    model = music_generator.LSTMMusic(music_featurizer._NUM_FEATURES, OUTPUT_SIZE_PITCH_SPACE, 
                                      OUTPUT_SIZE_QUARTER_LENGTH, HIDDEN_SIZE, NUM_LAYERS).to(device)
    if RETRAIN:
        model.load_state_dict(torch.load("music_sequencer_1.pth"))
    loss_fn_pitch_space = nn.CrossEntropyLoss()
    loss_fn_quarter_length = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    NUM_EPOCHS = 1200
    BATCH_SIZE = 100

    print(f"Training for {NUM_EPOCHS} epochs...")
    train_sequences(model, loss_fn_pitch_space, loss_fn_quarter_length, optimizer, X, 
                    TRAINING_SEQUENCE_MIN_LENGTH, TRAINING_SEQUENCE_MAX_LENGTH, BATCH_SIZE, 
                    NUM_EPOCHS, 10, 100, "music_sequencer_1.pth", device=device)
    
    # Save the model state
    # torch.save(model.state_dict(), "music_sequencer.pth")
