"""
File: music_finder.py

Finds MusicXML files and prepares them for training
"""

import music_featurizer
import music21
import os
import re
import torch


def find_files(directory_name: str) -> list:
    """
    Finds all MusicXML files within a directory (and its subdirectories, if recurse=True)
    :param directory_name: The directory name
    :return: A list of file names
    """
    files_music_xml = []
    search = re.compile(r"(\.xml$)|(\.musicxml$)|(\.mxl$)")

    for path, subdirectories, files in os.walk(directory_name):
        for name in files:
            result = search.search(name)
            if result:
                files_music_xml.append(os.path.join(path, name))

    return files_music_xml


def prepare_m21_corpus(composer_name: str, device: str):
    """
    Prepares all of the XML files in a music21 corpus for processing, and tokenizes them into sequences
    :param composer_name: The name of the composer to process
    :param device: The device for the tensors
    :return: The X and (y tuple) items
    """
    TEMPO_DICT = {1: 100}
    TRAINING_SEQUENCE_MAX_LENGTH = 15
    TRAINING_SEQUENCE_MIN_LENGTH = 3
    files = music21.corpus.getComposer(composer_name)
    files_xml = filter(lambda x: ".xml" in str(x), files)
    X = []
    y_pitch_space = []
    y_quarter_length = []

    # Prepare the data for running through the model. We want sequences of length N for training.
    for file in files_xml:
        score = music21.corpus.parse(file)

        for i in music_featurizer.get_staff_indices(score):
            data = music_featurizer.load_data(score[i], TEMPO_DICT)
            data = music_featurizer.tokenize(data, False)
            data_x, data_y = music_featurizer.make_sequences(data, TRAINING_SEQUENCE_MAX_LENGTH, device=device)
            X.append(data_x)
            y_pitch_space.append(data_y[0])
            y_quarter_length.append(data_y[1])

    X = torch.vstack(X)
    y_pitch_space = torch.vstack(y_pitch_space)
    y_quarter_length = torch.vstack(y_quarter_length)

    return X, y_pitch_space, y_quarter_length


def prepare_directory(directory_name: str, device: str):
    """
    Prepares all of the XML files in a directory for processing, and tokenizes them into sequences
    :param directory_name: The name of the directory to process
    :param device: The device for the tensors
    :return: The X and (y tuple) items
    """
    TEMPO_DICT = {1: 100}
    TRAINING_SEQUENCE_LENGTH = 10
    files = find_files(directory_name)
    X = []
    y_pitch_space = []
    y_quarter_length = []

    # Prepare the data for running through the model. We want sequences of length N for training.
    for file in files:
        score = music21.converter.parse(file)

        for i in music_featurizer.get_staff_indices(score):
            data = music_featurizer.load_data(score[i], TEMPO_DICT)
            data = music_featurizer.tokenize(data, False)
            data_x, data_y = music_featurizer.make_sequences(data, TRAINING_SEQUENCE_LENGTH, device=device)
            X.append(data_x)
            y_pitch_space.append(data_y[0])
            y_quarter_length.append(data_y[1])

    X = torch.vstack(X)
    y_pitch_space = torch.vstack(y_pitch_space)
    y_quarter_length = torch.vstack(y_quarter_length)

    return X, y_pitch_space, y_quarter_length
