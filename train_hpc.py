"""
File: train_hpc.py

This file is modified to run on the University of Iowa Argon high-performance computing system.

This module trains the music sequence generator. You can either train a model from
scratch, or you can choose to continue training a model that was previously saved
to disk. The training function will output status messages and save periodically.
"""

import dataset
import datetime
import json
import feature_definitions
import featurizer
import model_definition
import os
from pathlib import Path
import torch
import torch.distributed
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader


def setup_parallel(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def train_sequences(model, dataloader, loss_fns, loss_weights, optimizer, num_epochs, status_interval, 
                    save_interval, model_metadata, device="cpu") -> None:
    """
    Trains the model. This training function expects a DataLoader which will feed it batches
    of sequences in randomized order. The DataLoader takes care of serving up labels as well.
    This function will output a routine status message with loss and estimated time remaining.
    The model state is routinely saved to disk, so you can use it while it is training.
         
    :param model: The model to train
    :param dataloader: The dataloader
    :param loss_fns: The loss functions
    :param loss_weights: The loss weights
    :param optimizer: The optimizer
    :param num_epochs: The number of epochs for training
    :param status_interval: How often (in epochs) to print an update message
    :param save_interval: Save to disk every N epochs
    :param model_metadata: The model metadata
    :param device: The device that is being used for training
    """

    training_stats = {
        "time": datetime.datetime.now(), 
        "total_time": datetime.timedelta(), 
        "last_epoch_duration": datetime.timedelta(),
        "seconds_remaining": 0.0,
        "average_loss_this_epoch": 0.0,
        "last_completed_epoch": 0,
        "total_epochs": num_epochs,
        "status_interval": status_interval
    }

    for epoch in range(num_epochs):        
        # Track the total loss and number of batches processed this epoch
        total_loss_this_epoch = 0
        num_batches_this_epoch = 0

        # Iterate through each batch in the dataloader. The batch will have 3 labels per sequence.
        for x, y1, y2, lengths in dataloader:
            optimizer.zero_grad()

            # Prepare for running through the net
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y = (y1, y2)
            hidden = model.init_hidden(x.shape[0])
            
            # This is necessary because of the size of the output labels.
            model.lstm.flatten_parameters()

            # Run the current batch through the net
            output, _ = model(x, lengths, hidden)

            # Compute loss
            loss = [loss_fns[i](output[i], y[i]) * loss_weights[i] for i in range(len(y))]
            total_loss = sum(loss)
            total_loss_this_epoch += total_loss.item()
            num_batches_this_epoch += 1
            
            # Update weights. Clip gradients to help avoid exploding and vanishing gradients.
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Generate status. The status consists of the epoch number, average loss, epoch completion
        # time, epoch duration (MM:SS), and estimated time remaining (HH:MM:SS).
        training_stats["last_completed_epoch"] = epoch
        time_new = datetime.datetime.now()
        training_stats["last_epoch_duration"] = time_new - training_stats["time"]
        training_stats["total_time"] += training_stats["last_epoch_duration"]
        training_stats["time"] = time_new
        training_stats["average_loss_this_epoch"] = round(total_loss_this_epoch / num_batches_this_epoch, 4)
        status_output(training_stats)

        # Save to disk if it is the right epoch interval
        if epoch % save_interval == save_interval - 1:
            model_metadata["loss"] = training_stats["average_loss_this_epoch"]
            with open(FILE_NAME, "w") as model_json_file:
                model_json_file.write(json.dumps(model_metadata))
            torch.save(model.state_dict(), model_metadata["state_dict"])
            print("Saved to disk\n")


def status_output(training_stats):
    """
    Outputs training status
    :param training_stats: A dictionary with training statistics
    """
    seconds_remaining = int((training_stats["total_time"].seconds / (training_stats["last_completed_epoch"] + 1)) * \
                            (training_stats["total_epochs"] - training_stats["last_completed_epoch"] - 1))
    
    status_message = "----------------------------------------------------------------------\n" + \
                     "epoch {0:<4}\nloss: {1:<6} | completion time: {2} | epoch duration (MM:SS): " + \
                     "{3:02}:{4:02}\nest. time remaining (HH:MM:SS): {5:02}:{6:02}:{7:02}\n"
    status_message = status_message.format(
        training_stats["last_completed_epoch"] + 1, training_stats["average_loss_this_epoch"], 
        training_stats["time"].strftime("%m-%d %H:%M:%S"), 
        training_stats["last_epoch_duration"].seconds // 60, 
        training_stats["last_epoch_duration"].seconds % 60, 
        seconds_remaining // (60 ** 2), 
        seconds_remaining // 60 % 60, 
        seconds_remaining % 60
    )
    
    # Output status
    if training_stats["status_interval"] is not None and \
        training_stats["last_completed_epoch"] % training_stats["status_interval"] == training_stats["status_interval"] - 1:
        print(status_message)


if __name__ == "__main__":
    MODEL_SUFFIX = "_8_13_24"                                   # The suffix for the current model
    ROOT_PATH = "/Users/jmartin50/source/music_generator"              # Specifies the absolute root directory
    PATH = os.path.join(ROOT_PATH, "data/train")                # The path to the training corpus
    FILE_NAME = os.path.join(ROOT_PATH, f"data/model{MODEL_SUFFIX}.json")    # The path to the model metadata JSON file
    RETRAIN = False                                             # Whether or not to continue training the same model
    NUM_EPOCHS = 500                                            # The number of epochs to train
    LEARNING_RATE = 0.001                                       # The model learning rate
    NUM_DATALOADER_WORKERS = 16                                 # The number of workers for the dataloader
    PRINT_UPDATE_INTERVAL = 1                                   # The epoch interval for printing training status
    MODEL_SAVE_INTERVAL = 10                                    # The epoch interval for saving the model
    JSON_CORPUS = os.path.join(ROOT_PATH, "data/corpus1.json")  # The JSON corpus to use

    # Make the train directory if it doesn't exist already
    Path(PATH).mkdir(parents=True, exist_ok=True)

    # The model metadata - save to JSON file
    model_metadata = {
        "model_name": "bach",
        "path": FILE_NAME,
        "training_sequence_min_length": 2,
        "training_sequence_max_length": 20,
        "num_layers": 8,
        "hidden_size": 1024,
        "batch_size": 1000,
        "state_dict": os.path.join(ROOT_PATH, f"data/music_sequencer{MODEL_SUFFIX}.pth"),
        "num_features": feature_definitions.NUM_FEATURES,
        "output_sizes": [len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING), len(feature_definitions.QUARTER_LENGTH_ENCODING)],
        "loss": None
    }
    with open(FILE_NAME, "w") as model_json_file:
        model_json_file.write(json.dumps(model_metadata))

    print("Loading dataset...\n")
    sequence_dataset = dataset.MusicXMLDataSet(featurizer.load_json_corpus(JSON_CORPUS), 
                                               model_metadata["training_sequence_min_length"], 
                                               model_metadata["training_sequence_max_length"])
    dataloader = DataLoader(sequence_dataset, model_metadata["batch_size"], True, 
                            collate_fn=dataset.MusicXMLDataSet.collate, num_workers=NUM_DATALOADER_WORKERS)
    print("Dataset loaded.\n")

    # Prefer CUDA if available, otherwise MPS (if on Apple), or CPU as a last-level default
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}\n")
    
    # Load and prepare the model. If retraining the model, we will need to load the
    # previous state dictionary so that we aren't training from scratch.
    model = model_definition.LSTMMusic(model_metadata["num_features"], model_metadata["output_sizes"], 
                                      model_metadata["hidden_size"], model_metadata["num_layers"], device).to(device)
    if RETRAIN:
        print(f"Retraining model from state dict {model_metadata['state_dict']}")
        model.load_state_dict(torch.load(model_metadata["state_dict"]))
    loss_fn = [nn.CrossEntropyLoss() for i in range(len(model_metadata["output_sizes"]))]
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_weights = torch.tensor([1.0, 1.0])  # emphasize the loss of the accidental
    loss_weights = loss_weights * loss_weights.numel() / torch.sum(loss_weights)   # normalize the loss weights

    # Train the model
    print(f"Training for {NUM_EPOCHS} epochs...\n")
    train_sequences(model, dataloader, loss_fn, loss_weights, optimizer, NUM_EPOCHS, PRINT_UPDATE_INTERVAL, MODEL_SAVE_INTERVAL, model_metadata, device)
    
