"""
Name: xml_gen.py
Author: Jeff Martin
Email: jeffreymartin@outlook.com
Date: 9/25/21

This file contains functions for creating Music21 scores and exporting them to MusicXML.
It is a copy of what is contained in the mutil library.
"""

import music21
import numpy
import xml.etree.ElementTree

from pctheory import pitch


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


def add_sequence(part, item_sequence, lyric_sequence=None, measure_no=1, bar_duration=4.0, offset=0):
    """
    Adds a sequence of notes or chords to a Part or PartStaff
    :param part: The Part or PartStaff
    :param item_sequence: A list of pitches, chords, or rests in order.
    :param lyric_sequence: A list of lyrics corresponding to the pitch durations. If the current index is a list, we can make multiple verses.
    :param measure_no: The measure number
    :param bar_duration: The quarter duration of the first measure
    :param offset: The offset position of the first note (default to 0)
    :return:
    """
    m = 0                                # The current measure index
    current_bar_duration = bar_duration  # The current measure duration in quarter notes
    current_offset = offset              # The offset for the next chord to insert

    # Find the starting measure index
    for i in range(len(part)):
        if type(part[i]) == music21.stream.Measure:
            if part[i].number == measure_no:
                m = i
    
    # Insert each item
    for i in range(len(item_sequence)):
        total_duration = item_sequence[i].duration.quarterLength      # The total duration of the chord
        remaining_duration = item_sequence[i].duration.quarterLength  # The remaining duration of the chord

        # Loop until the entire duration of the chord has been inserted
        while remaining_duration > 0:
            # The item to insert
            c = None
            
            # Get the duration of this fragment of the chord
            duration = 0
            if current_bar_duration - current_offset >= remaining_duration:
                duration = remaining_duration
            else:
                duration = current_bar_duration - current_offset

            if type(item_sequence[i]) == music21.chord.Chord:
                # Create the chord
                c = music21.chord.Chord(item_sequence[i].notes, quarterLength=duration)
            elif type(item_sequence[i]) == music21.note.Note:
                # Create the note
                c = music21.note.Note(nameWithOctave=item_sequence[i].nameWithOctave, quarterLength=duration)
            else:
                # Create the rest
                c = music21.note.Rest(quarterLength=duration)

            # Create ties and attach lyrics as appropriate
            if remaining_duration == total_duration and duration < total_duration:
                c.tie = music21.tie.Tie("start")
                if lyric_sequence is not None:
                    if type(lyric_sequence[i]) == list:
                        c.lyrics = [music21.note.Lyric(number=j, text=lyric_sequence[i][j]) for j in range(len(lyric_sequence[i]))]
                    else:
                        c.lyrics = [music21.note.Lyric(number=1, text=lyric_sequence[i])]
            elif total_duration > remaining_duration > duration:
                c.tie = music21.tie.Tie("continue")
            elif total_duration > remaining_duration == duration:
                c.tie = music21.tie.Tie("stop")
            else:
                if lyric_sequence is not None:
                    if type(lyric_sequence[i]) == list:
                        c.lyrics = [music21.note.Lyric(number=j + 1, text=lyric_sequence[i][j])
                                    for j in range(len(lyric_sequence[i]))]
                    else:
                        c.lyrics = [music21.note.Lyric(number=1, text=lyric_sequence[i])]

            # Insert the chord into the current measure
            part[m].insert(current_offset, c)

            # Update the offset and remaining duration
            current_offset += duration
            remaining_duration -= duration
            if current_offset >= current_bar_duration:
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


def add_measures(score, num=10, start_num=1, key=None, meter=None, initial_offset=0, padding_left=0, padding_right=0):
    """
    Adds measures to a score
    :param score: The score
    :param num: The number of measures to add
    :param start_num: The starting measure number
    :param key: The key signature for the first measure
    :param meter: The time signature for the first measure
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
            m = music21.stream.Measure(number=start_num, offset=initial_offset, paddingLeft=padding_left)
            if meter is not None:
                ts = music21.meter.TimeSignature(meter)
                m.insert(0, ts)
            if key is not None:
                m.insert(0, music21.key.KeySignature(key))
            duration = m.timeSignature.duration.quarterLength
            initial_offset += m.barDuration.quarterLength
            item.append(m)

            # Add the remaining measures.
            for i in range(1, num):
                # The final measure may be shorter to compensate for the pickup measure.
                if i == num - 1:
                    m1 = music21.stream.Measure(number=i + start_num, offset=initial_offset, paddingRight=padding_right)
                    item.append(m1)
                else:
                    # note: previous version provided the quarterLength of the measure here, but apparently
                    # support for that has been deprecated in music21, so I removed that specification.
                    m1 = music21.stream.Measure(number=i + start_num, offset=initial_offset)
                    item.append(m1)
                    initial_offset += duration


def cleanup_semi_closed(chord):
    """
    Cleans up music21 semiclosed positions
    :param chord:
    :return:
    """
    for item in chord.pitches:
        if item.name == "E#" or item.name == "B#" or item.name == "C-" or item.name == "F-":
            item.midi += 12
        elif item.accidental.name == "double-sharp" or item.accidental.name == "double-flat":
            item.midi += 12


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


def create_score_piano(title="Score", composer="Jeff Martin", num_measures: int=10, key=None, time_signature="4/4"):
    """
    Creates a piano score. This allows you to get started easily.
    :param title: The title of the score
    :param composer: The composer name
    :param num_measures: The number of measures to add
    :param key: The key to use (if None, will be C)
    :param time_signature: The time signature to use
    :return: A piano score
    """
    s = music21.stream.Score()
    s.insert(0, music21.metadata.Metadata())
    s.metadata.title = title
    s.metadata.composer = composer
    add_instrument_multi(s, "Piano", "Pno.", 2)
    if key is None:
        key = 0
    add_measures(s, num_measures, 1, key, time_signature)
    add_item(s[1], music21.clef.TrebleClef(), 1)
    add_item(s[2], music21.clef.BassClef(), 1)
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
                    m_list.append(music21.chord.Chord([j + 60 for j in current_item], quarterLength=current_duration))
                elif type(current_item[0]) == pitch.Pitch:
                    m_list.append(music21.chord.Chord([music21.pitch.Pitch(p.p / (p.mod / 12) + 60) for p in current_item], quarterLength=current_duration))
            
            elif type(current_item) == float or type(current_item) == int:
                if current_item == -numpy.inf:
                    m_list.append(music21.note.Rest(quarterLength=current_duration))
                else:
                    m_list.append(music21.note.Note(current_item + 60, quarterLength=current_duration))
            
            elif type(current_item) == pitch.Pitch:
                m_list.append(music21.note.Note(music21.pitch.Pitch(items[i].p / (items[i].mod / 12) + 60), quarterLength=current_duration))
    return m_list


def make_semi_closed(chord):
    """
    Makes a chord semi-closed and cleans up the notation. This is useful if you made 
    a chord directly from a PitchClass set without adjusting the spacing. It uses
    the music21 semiClosedPosition algorithm to space the chord, and then cleans up
    some common mistakes this algorithm makes with accidentals.
    :param chord: A chord
    :return: A semi-closed position rendering
    """
    chord.semiClosedPosition(inPlace=True)
    for item in chord.pitches:
        if item.name == "E#" or item.name == "B#" or item.name == "C-" or item.name == "F-":
            item.midi += 12
        elif item.accidental.name == "double-sharp" or item.accidental.name == "double-flat":
            item.midi += 12
        if item.midi > 84:
            item.midi -= 12


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
    

def split_pset_for_grand_staff(chord):
    """
    Splits a Pitch set for distribution across a grand staff
    :param chord: A set or list of pitches
    :return: Two lists of pitches, one for each staff. The first list is for the top staff,
    and the second list is for the bottom staff.
    """
    bottom_staff = []
    top_staff = []
    for pitch in chord:
        if pitch.p < 0:
            bottom_staff.append(pitch)
        else:
            top_staff.append(pitch)
    return top_staff, bottom_staff


def test():
    """
    Tests the xml_gen code
    :return:
    """
    s = create_score()
    add_instrument_multi(s, "Piano", "Pno.", 2, "brace", True)
    add_measures(s, 5, 0, 3, "3/4", 1, 1)
    add_item(s[1], music21.clef.TrebleClef(), 0, 0)
    add_item(s[2], music21.clef.BassClef(), 0, 0)
    s[1][0].append(music21.note.Note("C#4", offset=2, quarterLength=1))
    s[1][1].append(music21.note.Note("D4", offset=0, quarterLength=1))
    s[1][1].append(music21.note.Note("C#4", offset=1, quarterLength=1))
    s[1][1].append(music21.note.Note("F#4", offset=2, quarterLength=1))
    s[1][2].append(music21.note.Note("A4", offset=0, quarterLength=1))
    s[1][2].append(music21.note.Note("E4", offset=1, quarterLength=1))
    s[1][2].append(music21.note.Note("B3", offset=2, quarterLength=1))
    s[1][3].append(music21.note.Note("G#4", offset=0, quarterLength=1))
    s[1][3].append(music21.note.Note("F#4", offset=1, quarterLength=1))
    s[1][3].append(music21.note.Note("E4", offset=2, quarterLength=1))
    s[1][4].append(music21.note.Note("C#4", offset=0, quarterLength=2))

    s.show()
