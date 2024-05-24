"""
File: music_train.py

This module trains the music sequence generator. You can either train a model from
scratch, or you can choose to continue training a model that was previously saved
to disk. The training function will output status messages and save periodically.
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


def train_sequences(model, loss_fn_pc, loss_fn_octave, loss_fn_quarter_length, optimizer, dataloader, 
                    num_epochs, status_interval, save_interval, model_state_file, device="cpu"):
    """
    Trains the model. This training function expects a DataLoader which will feed it batches
    of sequences in randomized order. The DataLoader takes care of serving up labels as well.
    This function will output a routine status message with loss and estimated time remaining.
    The model state is routinely saved to disk, so you can use it while it is training.
         
    :param model: The model to train
    :param loss_fn_pc: The loss function for pitch class
    :param loss_fn_octave: The loss function for octave
    :param loss_fn_quarter_length: The loss function for quarter length
    :param optimizer: The optimizer
    :param dataloader: The dataloader
    :param num_epochs: The number of epochs for training
    :param status_interval: How often (in epochs) to print an update message
    :param save_interval: Save to disk every N epochs
    :param model_state_file: The file name (for saving to disk)
    :param device: The device that is being used for the hidden matrices
    """

    t = datetime.datetime.now()
    total_time = datetime.timedelta()

    for epoch in range(num_epochs):        
        # Track the total loss and number of batches processed this epoch
        total_loss_this_epoch = 0
        num_batches_this_epoch = 0

        # Iterate through each batch in the dataloader. The batch will have 3 labels per sequence.
        for x, y1, y2, y3, lengths in dataloader:
            optimizer.zero_grad()

            # Prepare for running through the net
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)
            hidden = model.init_hidden(x.shape[0])

            # Run the current batch through the net
            output, _ = model(x, lengths, hidden)

            # Compute loss
            loss_pc = loss_fn_pc(output[0], y1)
            loss_octave = loss_fn_octave(output[1], y2)
            loss_quarter_length = loss_fn_quarter_length(output[2], y3)
            total_loss = loss_pc + loss_octave + loss_quarter_length
            total_loss_this_epoch += total_loss.item()
            num_batches_this_epoch += 1
            
            # Update weights
            total_loss.backward()
            optimizer.step()
        
        # Output status. The status consists of the epoch number, average loss, epoch completion
        # time, epoch duration (MM:SS), and estimated time remaining (HH:MM:SS).
        if epoch % status_interval == status_interval - 1:
            time_new = datetime.datetime.now()
            delta = time_new - t
            total_time += delta
            t = time_new
            seconds_remaining = int((total_time.seconds / (epoch + 1)) * (num_epochs - epoch - 1))
            print("epoch {0:<4}\nloss: {1:<6} | completion time: {2} | duration (MM:SS): {3:02}:{4:02}\n"
                  "est. time remaining (HH:MM:SS): {5:02}:{6:02}:{7:02}\n".format(
                    epoch+1, round(total_loss_this_epoch / num_batches_this_epoch, 4), t.strftime("%m-%d %H:%M:%S"), 
                    delta.seconds // 60, delta.seconds % 60, 
                    seconds_remaining // (60 ** 2), seconds_remaining // 60 % 60, seconds_remaining % 60
                )
            )

        # Save to disk if it is the right epoch interval
        if epoch % save_interval == save_interval - 1:
            torch.save(model.state_dict(), model_state_file)


if __name__ == "__main__":
    #######################################################################################
    # YOU WILL NEED TO EDIT THIS MANUALLY
    #######################################################################################
    
    PATH = "./data/train"            # The path to the training corpus
    FILE_NAME = "./data/model.json"  # The path to the model metadata JSON file
    RETRAIN = False                  # Whether or not to continue training the same model
    NUM_EPOCHS = 200                 # The number of epochs to train
    LEARNING_RATE = 0.001            # The model learning rate
    
    # The model metadata - save to JSON file
    model_metadata = {
        "model_name": "bach",
        "training_sequence_min_length": 2,
        "training_sequence_max_length": 40,
        "num_layers": 8,
        "hidden_size": 2048,
        "batch_size": 400,
        "state_dict": "./data/music_sequencer_4.pth",
        "num_features": music_featurizer._NUM_FEATURES,
        "output_size_pc": len(music_featurizer._PS_ENCODING),
        "output_size_octave": len(music_featurizer._OCTAVE_ENCODING),
        "output_size_quarter_length": len(music_featurizer._QUARTER_LENGTH_ENCODING)
    }
    with open(FILE_NAME, "w") as model_json_file:
        model_json_file.write(json.dumps(model_metadata))

    # Get the corpus and prepare it as a dataset
    # files = music_finder.prepare_directory(PATH, device)
    files = music_finder.get_m21_corpus('bach')
    
    #######################################################################################
    # YOU PROBABLY DON'T NEED TO EDIT ANYTHING BELOW HERE
    #######################################################################################
    
    # Prefer CUDA if available, otherwise MPS (if on Apple), or CPU as a last-level default
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}")
    
    dataset = music_featurizer.MusicXMLDataSet(files, model_metadata["training_sequence_min_length"], 
                                               model_metadata["training_sequence_max_length"])
    dataloader = DataLoader(dataset, model_metadata["batch_size"], True, collate_fn=music_featurizer.MusicXMLDataSet.collate, num_workers=8)
        
    # Load and prepare the model. If retraining the model, we will need to load the
    # previous state dictionary so that we aren't training from scratch.
    model = music_generator.LSTMMusic(model_metadata["num_features"], model_metadata["output_size_pc"], model_metadata["output_size_octave"], 
                                      model_metadata["output_size_quarter_length"], model_metadata["hidden_size"], 
                                      model_metadata["num_layers"], device).to(device)
    if RETRAIN:
        model.load_state_dict(torch.load(model_metadata["state_dict"]))
    loss_fn_pc = nn.CrossEntropyLoss()
    loss_fn_octave = nn.CrossEntropyLoss()
    loss_fn_quarter_length = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print(f"Training for {NUM_EPOCHS} epochs...")
    train_sequences(model, loss_fn_pc, loss_fn_octave, loss_fn_quarter_length, optimizer, dataloader, 
                    NUM_EPOCHS, 1, 10, model_metadata["state_dict"], device=device)
    