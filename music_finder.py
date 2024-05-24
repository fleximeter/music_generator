"""
File: music_finder.py

This module finds MusicXML files and prepares them for training. It can find files
in a music21 corpus or in a directory and its subfolders.
"""

import music21
import os
import re


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


def get_m21_corpus(composer_name: str) -> list:
    """
    Finds all of the XML files in a music21 corpus
    :param composer_name: The name of the composer to process
    :return: The files
    """
    files = music21.corpus.getComposer(composer_name)
    files_xml = filter(lambda x: ".xml" in str(x), files)
    return files_xml
