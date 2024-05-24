# Music Generator

## About
This repository is a generative AI system for producing the next note in a sequence of notes. It makes melodies. It can train models on a collection of MusicXML files, including `music21` corpuses. It is based on the PyTorch LSTM architecture, and will need considerable resources for training.

## Resource needs
The default training device is CUDA, and MPS is the first fallback, with a CPU as the last-level default. When training, the estimated time remaining is output, which will help you determine if you are making your computer do too much work.

## Training a model
To train a model, you run the `music_train.py` program. You will need to specify the location to save the model metadata dictionary, as well as items in the dictionary (hyperparameters, etc.) While training, the model will routinely save its state to a file, specified in the model metadata dictionary. Once the model has saved its state for the first time, you can start making predictions as the model continues to train and periodically updates its state file.

## Making predictions
To make predictions, you run the `music_predict.py` program. You will need to specifiy the location of the model metadata file, provide a MusicXML prompt file, and specify the number of notes to generate. The predictor will automatically open the generated music in your default `music21` score viewer.

## Dependencies
You will need to install the following packages to use this repository:
`music21`, `numpy`, `pytorch`, `pytorch-cuda` (if you are running on CUDA), `regex`, `sendgrid` (for sending training status updates)

To install on a Python virtualenv, run `pip install music21 numpy pytorch pytorch-cuda regex sendgrid`
