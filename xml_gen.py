"""
Name: xml_gen.py
Author: Jeff Martin
Email: jeffreymartin@outlook.com
Date: 9/25/21

This file contains functions for creating Music21 scores and exporting them to MusicXML.
It is a copy of what is contained in the mutil library with some modifications for
handling actual MIDI note numbers rather than Morris p-space numbers.
"""

from fractions import Fraction
import math
import music21
import numpy as np
import xml.etree.ElementTree


def add_item(part, item, measure_no, offset=0):
    """
    Adds an item such as a Clef, KeySignature, or TimeSignature to a Part or PartStaff
    :param part: The Part or PartStaff
    :param item: The item (can be anything, including Clef, KeySignature, TimeSignature, Note, Rest, etc.
    :param measure_no: The measure number
    :param offset: The offset position (default to 0)
    :return:
    """
    for stream_item in part:
        if type(stream_item) == music21.stream.Measure:
            if stream_item.number == measure_no:
                # Find the index at which to insert the item.
                index = len(stream_item)
                for i in range(len(stream_item)):
                    if stream_item[i].offset >= offset:
                        index = i
                        break
                stream_item.insert(index, item)


def add_sequence(part, item_sequence, lyric_sequence=None, measure_no=1, bar_duration=4.0):
    """
    Adds a sequence of notes or chords to a Part or PartStaff
    :param part: The Part or PartStaff
    :param item_sequence: A list of pitches, chords, or rests in order.
    :param lyric_sequence: A list of lyrics corresponding to the pitch durations. If the current index is a list, we can make multiple verses.
    :param measure_no: The measure number
    :param bar_duration: The quarter duration of the first measure
    :return:
    """
    m = 0                                # The current measure index
    current_bar_duration = bar_duration  # The current measure duration in quarter notes
    current_offset = 0                   # The offset for the next chord to insert

    # Find the starting measure index
    for i in range(len(part)):
        if type(part[i]) == music21.stream.Measure:
            if part[i].number == measure_no:
                m = i
    
    # Insert each item
    for i, current_item in enumerate(item_sequence):
        total_duration = item_sequence[i].duration.quarterLength      # The total duration of the chord
        remaining_duration = item_sequence[i].duration.quarterLength  # The remaining duration of the chord

        # Loop until the entire duration of the chord has been inserted
        while remaining_duration > 0:
            # The item to insert
            new_item_to_insert = None
            
            # Get the duration of this fragment of the chord
            new_item_duration = 0
            if current_bar_duration - part[m].paddingLeft - current_offset >= remaining_duration:
                new_item_duration = remaining_duration
            else:
                new_item_duration = current_bar_duration - part[m].paddingLeft - current_offset

            # CREATE NOTE, CHORD, OR REST
            if type(current_item) == music21.chord.Chord:
                new_item_to_insert = music21.chord.Chord(current_item.notes, quarterLength=new_item_duration)
            elif type(current_item) == music21.note.Note:
                new_item_to_insert = music21.note.Note(nameWithOctave=current_item.nameWithOctave, quarterLength=new_item_duration)
            else:
                new_item_to_insert = music21.note.Rest(quarterLength=new_item_duration)

            # TIES AND LYRICS: Create ties and attach lyrics as appropriate
            if remaining_duration == total_duration and new_item_duration < total_duration:
                new_item_to_insert.tie = music21.tie.Tie("start")
                if lyric_sequence is not None:
                    if type(lyric_sequence[i]) == list:
                        new_item_to_insert.lyrics = [music21.note.Lyric(number=j, text=lyric_sequence[i][j]) for j in range(len(lyric_sequence[i]))]
                    else:
                        new_item_to_insert.lyrics = [music21.note.Lyric(number=1, text=lyric_sequence[i])]
            elif total_duration > remaining_duration > new_item_duration:
                new_item_to_insert.tie = music21.tie.Tie("continue")
            elif total_duration > remaining_duration == new_item_duration:
                new_item_to_insert.tie = music21.tie.Tie("stop")
            else:
                if lyric_sequence is not None:
                    if type(lyric_sequence[i]) == list:
                        new_item_to_insert.lyrics = [music21.note.Lyric(number=j + 1, text=lyric_sequence[i][j])
                                    for j in range(len(lyric_sequence[i]))]
                    else:
                        new_item_to_insert.lyrics = [music21.note.Lyric(number=1, text=lyric_sequence[i])]

            # Insert the chord into the current measure
            part[m].insert(current_offset, new_item_to_insert)

            # Update the offset and remaining duration
            current_offset += new_item_duration
            remaining_duration -= new_item_duration
            if current_offset >= current_bar_duration - part[m].paddingLeft:
                m += 1
                current_offset = 0
                if m < len(part) and part[m].timeSignature is not None:
                    current_bar_duration = part[m].barDuration.quarterLength


def add_instrument(score, name, abbreviation):
    """
    Adds a violin to the score
    :param score: The score
    :param name: The name of the instrument
    :param abbreviation: The abbreviation of the instrument
    :return:
    """
    instrument = music21.stream.Part(partName=name, partAbbreviation=abbreviation)
    score.append(instrument)


def add_instrument_multi(score, name, abbreviation, num_staves, symbol="brace", bar_together=True):
    """
    Adds a part with multiple staves (such as a piano) to a score
    :param score: The score
    :param name: The name of the instrument
    :param abbreviation: The abbreviation of the instrument
    :param num_staves: The number of staves for the instrument
    :param symbol: The bracket or brace style for the instrument
    :param bar_together: Whether or not to bar the staves together
    :return:
    """
    staff_list = []
    for i in range(num_staves):
        staff_list.insert(i, music21.stream.PartStaff(partName=name + str(i + 1)))
    grp = music21.layout.StaffGroup(staff_list, name=name, abbreviation=abbreviation, symbol=symbol,
                                    barTogether=bar_together)
    for staff in staff_list:
        score.append(staff)
    score.append(grp)


def add_measures(score, num=10, start_num=1, key=None, meter=None, bar_duration=4.0, initial_offset=0.0, padding_left=0.0, padding_right=0.0):
    """
    Adds measures to a score
    :param score: The score
    :param num: The number of measures to add
    :param start_num: The starting measure number
    :param key: The key signature for the first measure
    :param meter: The time signature for the first measure
    :param bar_duration: The duration for each measure
    :param initial_offset: The offset for the first measure
    :param padding_left: Makes the first measure a pickup measure. This is the number of beats to subtract.
    :param padding_right: Makes the last measure shorter. This is the number of beats to subtract.
    :return:
    """
    for item in score:
        # We only add measures to streams that are Parts or PartStaffs.
        if type(item) == music21.stream.Part or type(item) == music21.stream.PartStaff:
            # The first measure gets special treatment. It may be given a time signature, a key signature,
            # and may be a pickup measure.
            m = music21.stream.Measure(number=start_num)
            if meter is not None:
                m.timeSignature = music21.meter.TimeSignature(meter)
            m.paddingLeft = padding_left
            if key is not None:
                m.insert(0, music21.key.KeySignature(key))
            m.duration = music21.duration.Duration(bar_duration)
            item.insert(initial_offset, m)
            initial_offset += bar_duration - padding_left
            
            # Add the remaining measures.
            for i in range(1, num):
                # The final measure may be shorter to compensate for the pickup measure.
                # note: previous version provided the quarterLength of the measure here, but apparently
                # support for that has been deprecated in music21, so I removed that specification.
                m1 = music21.stream.Measure(number=i + start_num)
                m1.duration = music21.duration.Duration(bar_duration)
                if i == num - 1:
                    m.paddingRight = padding_right
                item.insert(initial_offset, m1)
                initial_offset += bar_duration


def create_score(title="Score", composer="Jeff Martin"):
    """
    Creates a score
    :param title: The title of the score
    :param composer: The composer name
    :return: A score
    """
    s = music21.stream.Score()
    s.insert(0, music21.metadata.Metadata())
    s.metadata.title = title
    s.metadata.composer = composer
    return s


def export_to_xml(score, path):
    """
    Exports a score to a MusicXML file
    :param score: The score
    :param path: The path
    :return:
    """
    converter = music21.musicxml.m21ToXml.ScoreExporter(score)
    output = xml.etree.ElementTree.tostring(converter.parse())
    with open(path, "wb") as file:
        file.write(output)


def make_music21_list(items, durations):
    """
    Makes a music21 list
    :param items: A list of items
    :param durations: A list of quarter durations
    :return: A list of music21 items
    """
    m_list = []
    if len(items) == len(durations):
        for i in range(len(items)):
            current_item = items[i]
            current_duration = durations[i]
            
            # Handles chords            
            if type(current_item) == set:
                current_item = list(current_item)
            if type(current_item) == list:
                if len(current_item) == 0:
                    m_list.append(music21.note.Rest(current_duration))
                elif type(current_item[0]) == int or type(current_item[0]) == float:
                    m_list.append(music21.chord.Chord(current_item, quarterLength=current_duration))
                
            # Handles notes
            elif type(current_item) == tuple and len(current_item) >= 2:
                note = music21.note.Note()
                note.step = current_item[0]
                note.octave = current_item[1]
                note.quarterLength = current_duration
                if len(current_item) == 3 and current_item[-1] != "None":
                    note.pitch.accidental = current_item[2]
                m_list.append(note)
            
            # Handles rests
            elif np.isneginf(current_item):
                m_list.append(music21.note.Rest(quarterLength=current_duration))
    
    return m_list


def remove_empty_measures(score):
    """
    Removes empty measures from a score. For a measure to be empty, 
    it must have no entries (notes, chords, or rests) in any voice or part.
    :param score: A score to remove empty measures from
    """
    non_empty_measures = set()

    # Find all measures that are definitely not empty
    for item in score:
        if type(item) == music21.stream.Part or type(item) == music21.stream.PartStaff:
            for item2 in item:
                if type(item2) == music21.stream.Measure:
                    if len(item2) > 0:
                        non_empty_measures.add(item2.number)

    # Remove all other measures
    for item in score:
        if type(item) == music21.stream.Part or type(item) == music21.stream.PartStaff:
            i = 0
            while i < len(item):
                if type(item[i]) == music21.stream.Measure:
                    if item[i].number not in non_empty_measures:
                        del item[i]
                        i -= 1
                i += 1

    # Set barline for final measure
    for part in score:
        if type(part) == music21.stream.Part or type(part) == music21.stream.PartStaff:
            for i in range(len(part) - 1, -1, -1):
                if type(part[i]) == music21.stream.Measure:
                    part[i].rightBarline = music21.bar.Barline("final")
                    break

