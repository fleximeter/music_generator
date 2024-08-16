# Music Generator

## About
This repository is a generative AI system for producing the next note in a sequence of notes (a melody). It can train models on a collection of MusicXML files, including `music21` corpuses. The architecture is the PyTorch LSTM architecture, and it will likely need considerable resources for training if you want an acceptable model. Predictions generate two classes for each instance: (letter name + accidental name + octave, as one unified class), and length in quarter notes. Learning accidentals and letter names helps the model to learn to use the appropriate accidentals for a given key.

## Resource needs
The default training device is CUDA, and MPS is the first fallback, with a CPU as the last-level default. When training, the estimated time remaining is output. This helps with gauging resource consumption.

## Setup
You will need to install the following packages to use this repository:
`music21`, `numpy`, `torch`, and `regex`

Visit https://pytorch.org/get-started/locally/ for PyTorch installation instructions (this is a good idea if you want to use CUDA).

After installing music21, you need to configure it to open MusicXML files with a score viewer like Sibelius or MuseScore. You can do this by running the command `python3 -m music21.configure` (see https://www.music21.org/music21docs/installing/installWindows.html for more details).

## Training a model
The simplest way to train a model is to use a music21 corpus. Install the dependencies listed above, then follow these steps:
1. Run the `save_data.py` program to generate a JSON corpus file that can be used in the training program.
2. Run the `train.py` program. NOTE: Before running, make sure that file locations are correctly specified. You might not need to do anything, but the location of the corpus JSON is specified in the code, and if it's located somewhere else, or has a different name, you would need to change that.

While training, the model will routinely save its state to a file, specified in the model metadata dictionary. Once the model has saved its state for the first time, you can start making predictions as the model continues to train and periodically updates its state file.

## Making predictions
To make predictions, you run the `music_predict.py` program. You will need to specifiy the location of the model metadata file, provide a MusicXML prompt file, and specify the number of notes to generate. The predictor will automatically open the generated music in your default `music21` score viewer.

Prompt files should only include single-voice melodic prompts, since this model is only designed to predict the next note in a melody.

## File descriptions
`corpus.py` - Contains functionality for loading all XML files in a given directory and parsing them with `music21`, as well as loading files from the `music21` corpus by composer.

`dataset.py` - Contains the definition for a `torch.utils.data.Dataset` subclass, `MusicXMLDataSet`, that imports `music21.stream.Score` objects and converts them to sequences that can be fed into a model.

`feature_definitions.py` - Defines features that can be extracted from a sequence of notes, and defines their one-hot representations.

`featurizer.py` - Contains functionality for featurizing scores, turning them into one-hot encodings, and processing predicted labels.

`model_definition.py` - Contains the model definition.

`predictor.py` - Contains functionality for making predictions based on a sequence of notes in a MusicXML score and a given model.

`tests.py` - Contains testing functionality

`train.py` - Contains functionality for training models.

`train_hpc.py` - A modified version of `train.py` for running on the University of Iowa Argon high-performance computing system.

`xml_gen.py` - Contains an interface for working with `music21` and easily generating `music21.stream.Score` objects.
