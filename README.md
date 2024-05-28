# Music Generator

## About
This repository is a generative AI system for producing the next note in a sequence of notes (a melody). It can train models on a collection of MusicXML files, including `music21` corpuses. The architecture is the PyTorch LSTM architecture, and it will likely need considerable resources for training if you want an acceptable model. Predictions generate two classes for each instance: (letter name + accidental name + octave, as one unified class), and length in quarter notes. Learning accidentals and letter names helps the model to learn to use the appropriate accidentals for a given key.

## Resource needs
The default training device is CUDA, and MPS is the first fallback, with a CPU as the last-level default. When training, the estimated time remaining is output. This helps with gauging resource consumption.

## Training a model
To train a model, you run the `music_train.py` program. You will need to specify the location to save the model metadata dictionary, as well as items in the dictionary (hyperparameters, etc.) While training, the model will routinely save its state to a file, specified in the model metadata dictionary. Once the model has saved its state for the first time, you can start making predictions as the model continues to train and periodically updates its state file.

## Making predictions
To make predictions, you run the `music_predict.py` program. You will need to specifiy the location of the model metadata file, provide a MusicXML prompt file, and specify the number of notes to generate. The predictor will automatically open the generated music in your default `music21` score viewer.

## Dependencies
You will need to install the following packages to use this repository:
`music21`, `numpy`, `pytorch`, `pytorch-cuda` (if you are running on CUDA), `regex`, `sendgrid` (for sending training status updates)

To install on a Python virtualenv, run `pip install music21 numpy pytorch pytorch-cuda regex sendgrid`

## File descriptions
`corpus.py` - Contains functionality for loading all XML files in a given directory and parsing them with `music21`, as well as loading files from the `music21` corpus by composer.

`dataset.py` - Contains the definition for a `torch.utils.data.Dataset` subclass, `MusicXMLDataSet`, that imports `music21.stream.Score` objects and converts them to sequences that can be fed into a model.

`feature_definitions.py` - Defines features that can be extracted from a sequence of notes, and defines their one-hot representations.

`featurizer.py` - Contains functionality for featurizing scores, turning them into one-hot encodings, and processing predicted labels.

`model_definition.py` - Contains the model definition.

`predictor.py` - Contains functionality for making predictions based on a sequence of notes in a MusicXML score and a given model.

`tests.py` - Contains testing functionality

`train.py` - Contains functionality for training models.

`xml_gen.py` - Contains an interface for working with `music21` and easily generating `music21.stream.Score` objects.
