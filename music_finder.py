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
    Tokenizes all of the XML files in a music21 corpus
    :param composer_name: The name of the composer to process
    :param device: The device for the tensors
    :return: The X items
    """
    TEMPO_DICT = {1: 100}
    files = music21.corpus.getComposer(composer_name)
    files_xml = filter(lambda x: ".xml" in str(x), files)
    X = []

    # Prepare the data for running through the model. We want sequences of length N for training.
    for file in files_xml:
        score = music21.corpus.parse(file)

        for i in music_featurizer.get_staff_indices(score):
            data = music_featurizer.load_data(score[i], TEMPO_DICT)
            data = music_featurizer.tokenize(data, False)
            X.append(data)
    
    X = torch.vstack(X).to(device)
    return X


def prepare_directory(directory_name: str, device: str):
    """
    Tokenizes all of the XML files in a directory
    :param directory_name: The name of the directory to process
    :param device: The device for the tensors
    :return: The X items
    """
    TEMPO_DICT = {1: 100}
    files = find_files(directory_name)
    X = []
    
    # Prepare the data for running through the model. We want sequences of length N for training.
    for file in files:
        score = music21.corpus.parse(file)

        for i in music_featurizer.get_staff_indices(score):
            data = music_featurizer.load_data(score[i], TEMPO_DICT)
            data = music_featurizer.tokenize(data, False)
            X.append(data)
    
    X = torch.vstack(X).to(device)
    return X
