"""
File: corpus.py

This module finds MusicXML files and prepares them for training. It can find files
in a music21 corpus or in a directory and its subfolders.
"""

import music21
import os
import re


def find_scores(directory_name: str) -> list:
    """
    Finds all MusicXML files within a directory (and its subdirectories, if recurse=True).
    Parses them as scores, and returns a list of scores.
    :param directory_name: The directory name
    :return: A list of file names
    """
    files_music_xml = []
    search = re.compile(r"(\.xml$)|(\.musicxml$)|(\.mxl$)")

    for path, subdirectories, files in os.walk(directory_name):
        for name in files:
            result = search.search(name)
            if result:
                files_music_xml.append(music21.converter.parse(os.path.join(path, name)))

    return files_music_xml


def get_m21_corpus(composer_name: str) -> list:
    """
    Finds all of the XML files in a music21 corpus
    :param composer_name: The name of the composer to process
    :return: The files
    """
    files = music21.corpus.getComposer(composer_name)
    files_xml = list(filter(lambda x: ".xml" in str(x), files))
    files_xml = [music21.corpus.parse(file) for file in files_xml]
    return files_xml
