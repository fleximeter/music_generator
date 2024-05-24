"""
File: music_train.py

This module trains the music sequence generator.
"""

import datetime
import json
import music_featurizer
import music_finder
import music_generator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_sequences(model, loss_fn_pitch_space, loss_fn_quarter_length, optimizer, dataloader, num_epochs, status_num=5, save_every=100, file="music_sequencer_1.pth", device="cpu"):
    """
    Trains the model. This training function expects a batch of data X, which will be turned into 
    sequences of different lengths from min_length to max_length. The labels y are calculated as
    part of the sequencing process. Each epoch, all sequences of all lengths are processed once.
    :param model: The model to train
    :param loss_fn_pitch_space: The loss function for pitch space
    :param loss_fn_quarter_length: The loss function for quarter length
    :param optimizer: The optimizer
    :param dataloader: The dataloader
    :param num_epochs: The number of epochs for training
    :param status_num: How often (in epochs) to print an update message
    :param save_every: Save every N times to disk
    :param file: The file name (for saving to disk)
    :param device: The device that is being used for the hidden matrices
    """
    # Train for N epochs
    t = datetime.datetime.now()
    total_time = datetime.timedelta()
    for epoch in range(num_epochs):        
        # Predict sequences of different lengths. Each time the outer loop runs,
        # the sequence length will be different.

        avg_loss = 0
        num_batches = 0
        for x, y1, y2, lengths in dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            # lengths = lengths.to(device)
            hidden = model.init_hidden(x.shape[0], device=device)
            output, hidden = model(x, lengths, hidden)

            # Compute loss and update weights
            loss_pitch_space = loss_fn_pitch_space(output[0], y1)
            loss_quarter_length = loss_fn_quarter_length(output[1], y2)
            total_loss = loss_pitch_space + loss_quarter_length
            avg_loss += total_loss.item()
            num_batches += 1
            total_loss.backward()
            optimizer.step()
        
        # Output status
        if epoch % status_num == status_num - 1:
            time_new = datetime.datetime.now()
            delta = time_new - t
            total_time += delta
            t = time_new
            time_remaining = (total_time.seconds / (epoch + 1)) * (num_epochs - epoch - 1)
            print("epoch {0:<4} | loss: {1:<6} | completion time: {2} | duration: {3:02}:{4:02} | est. time remaining: {5:02}:{6:02}:{7:02}".format(
                epoch+1, round(avg_loss / num_batches, 4), t.strftime("%m-%d %H:%M:%S"), 
                delta.seconds // 60, delta.seconds % 60, time_remaining // (60 ** 2), 
                time_remaining // 60, time_remaining % 60))

        if epoch % save_every == save_every - 1:
            torch.save(model.state_dict(), file)


if __name__ == "__main__":
    PATH = "./data/train"
    OUTPUT_SIZE_PITCH_SPACE = len(music_featurizer._PS_ENCODING)
    OUTPUT_SIZE_QUARTER_LENGTH = len(music_featurizer._QUARTER_LENGTH_ENCODING)
    LEARNING_RATE = 0.001
    FILE_NAME = "model.json"
    model_data = {
        "corpus_name": "bach",
        "training_sequence_min_length": 2,
        "training_sequence_max_length": 40,
        "num_layers": 8,
        "hidden_size": 2048,
        "batch_size": 400,
        "state_dict": "music_sequencer_4.pth",
        "num_features": music_featurizer._NUM_FEATURES
    }

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}")
    
    # files = music_finder.prepare_directory(PATH, device)
    files = music_finder.get_m21_corpus('bach')
    dataset = music_featurizer.MusicXMLDataSet(files, model_data["training_sequence_min_length"], 
                                               model_data["training_sequence_max_length"])
    dataloader = DataLoader(dataset, model_data["batch_size"], True, collate_fn=music_featurizer.MusicXMLDataSet.collate, num_workers=8)
    
    # Whether or not to continue training the same model
    RETRAIN = False
    model = music_generator.LSTMMusic(model_data["num_features"], OUTPUT_SIZE_PITCH_SPACE, 
                                      OUTPUT_SIZE_QUARTER_LENGTH, model_data["hidden_size"], 
                                      model_data["num_layers"], device).to(device)
    if RETRAIN:
        model.load_state_dict(torch.load(model_data["state_dict"]))
    loss_fn_pitch_space = nn.CrossEntropyLoss()
    loss_fn_quarter_length = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    NUM_EPOCHS = 1000
    with open(FILE_NAME, "w") as model_json_file:
        model_json_file.write(json.dumps(model_data))
    print(f"Training for {NUM_EPOCHS} epochs...")
    train_sequences(model, loss_fn_pitch_space, loss_fn_quarter_length, optimizer, dataloader, 
                    NUM_EPOCHS, 1, 10, model_data["state_dict"], device=device)
    