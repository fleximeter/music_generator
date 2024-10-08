"""
File: tests.py

This module has tests to verify that feature encoding works properly.
"""

from fractions import Fraction
import feature_definitions
import featurizer
import music21


def compare_data(original_data, detokenized_data) -> None:
    """
    Compares original data and detokenized data to verify that the tokenizing/
    detokenizing process has not corrupted the data
    :param original_data: The original data
    :param detokenized_data: The detokenized data
    """
    # Check the detokenized data to make sure it is the same as the original data
    all_same = True
    for i in range(len(data)):
        same = True
        if original_data[i]["letter_accidental_octave_name"] != detokenized_data[i]["letter_accidental_octave_name"]:
            same = False
            all_same = False
        elif original_data[i]["letter_accidental_name"] != detokenized_data[i]["letter_accidental_name"]:
            same = False
            all_same = False
        elif original_data[i]["letter_name"] != detokenized_data[i]["letter_name"]:
            same = False
            all_same = False
        elif original_data[i]["accidental_name"] != detokenized_data[i]["accidental_name"]:
            same = False
            all_same = False
        elif original_data[i]["octave"] != detokenized_data[i]["octave"]:
            same = False
            all_same = False
        elif original_data[i]["pitch_class_id"] != detokenized_data[i]["pitch_class_id"]:
            same = False
            all_same = False
        elif original_data[i]["ps"] != detokenized_data[i]["ps"]:
            same = False
            all_same = False
        elif original_data[i]["key_signature"] != detokenized_data[i]["key_signature"]:
            same = False
            all_same = False
        elif original_data[i]["mode"] != detokenized_data[i]["mode"]:
            same = False
            all_same = False
        elif original_data[i]["quarterLength"] != detokenized_data[i]["quarterLength"]:
            same = False
            all_same = False
        elif original_data[i]["beat"] != detokenized_data[i]["beat"]:
            same = False
            all_same = False
        elif original_data[i]["time_signature"] != detokenized_data[i]["time_signature"]:
            same = False
            all_same = False
    
        if not same:
            print(f"Error in note {i}")
    
    if not all_same:
        print(f"Error in encoding/decoding process")
    else:
        print(f"SUCCESS: All encoding/decoding tests passed.")


def detokenize(tokens) -> list:
    """
    Detokenizes one-hot features and returns them to regular encoding
    :param tokens: A tensor of one-hot encoded instances
    :return: The regular encoding
    """
    i = (
        (0, len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING)), 
        (len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING)),
        (len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING)),
        (len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING)),
        (len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING) + len(feature_definitions.MELODIC_INTERVAL_ENCODING)),
        (len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING) + len(feature_definitions.MELODIC_INTERVAL_ENCODING), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING) + len(feature_definitions.MELODIC_INTERVAL_ENCODING) + len(feature_definitions.KEY_SIGNATURE_ENCODING)),
        (len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING) + len(feature_definitions.MELODIC_INTERVAL_ENCODING) + len(feature_definitions.KEY_SIGNATURE_ENCODING), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING) + len(feature_definitions.MELODIC_INTERVAL_ENCODING) + len(feature_definitions.KEY_SIGNATURE_ENCODING) + len(feature_definitions.MODE_ENCODING)),
        (len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING) + len(feature_definitions.MELODIC_INTERVAL_ENCODING) + len(feature_definitions.KEY_SIGNATURE_ENCODING) + len(feature_definitions.MODE_ENCODING), len(feature_definitions.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(feature_definitions.QUARTER_LENGTH_ENCODING) + len(feature_definitions.BEAT_ENCODING) + len(feature_definitions.PITCH_CLASS_ENCODING) + len(feature_definitions.MELODIC_INTERVAL_ENCODING) + len(feature_definitions.KEY_SIGNATURE_ENCODING) + len(feature_definitions.MODE_ENCODING) + len(feature_definitions.TIME_SIGNATURE_ENCODING)),
    )

    notes = []
    for j in range(tokens.size(0)):
        letter_accidental_octave = tokens[j, i[0][0]:i[0][1]]
        quarter_length = tokens[j, i[1][0]:i[1][1]]
        beat = tokens[j, i[2][0]:i[2][1]]
        pitch_class = tokens[j, i[3][0]:i[3][1]]
        melodic_interval = tokens[j, i[4][0]:i[4][1]]
        key_signature = tokens[j, i[5][0]:i[5][1]]
        mode = tokens[j, i[6][0]:i[6][1]]
        time_signature = tokens[j, i[7][0]:i[7][1]]
        letter_accidental_octave = feature_definitions.REVERSE_LETTER_ACCIDENTAL_OCTAVE_ENCODING[letter_accidental_octave.argmax().item()]
        quarter_length = feature_definitions.REVERSE_QUARTER_LENGTH_ENCODING[quarter_length.argmax().item()]
        beat = feature_definitions.REVERSE_BEAT_ENCODING[beat.argmax().item()]
        pitch_class = feature_definitions.REVERSE_PITCH_CLASS_ENCODING[pitch_class.argmax().item()]
        melodic_interval = feature_definitions.REVERSE_MELODIC_INTERVAL_ENCODING[melodic_interval.argmax().item()]
        key_signature = feature_definitions.REVERSE_KEY_SIGNATURE_ENCODING[key_signature.argmax().item()]
        mode = feature_definitions.REVERSE_MODE_ENCODING[mode.argmax().item()]
        time_signature = feature_definitions.REVERSE_TIME_SIGNATURE_ENCODING[time_signature.argmax().item()]
        letter_name, accidental, octave = letter_accidental_octave.split('|')
        if octave != "None":
            octave = int(float(octave))
            pitch_class = float(pitch_class)
            ps = pitch_class + (octave + 1) * 12
            key_signature = int(float(key_signature))
            quarter_length = Fraction(quarter_length)
            beat = Fraction(beat)
        note = {
            "letter_accidental_octave_name": letter_accidental_octave,
            "letter_accidental_name" : "|".join(letter_accidental_octave.split('|')[:-1]),
            "letter_name": letter_name,
            "accidental_name": accidental,
            "octave": octave,
            "pitch_class_id": pitch_class,
            "ps": ps,
            "key_signature": key_signature,
            "mode": mode,
            "quarterLength": quarter_length,
            "beat": beat,
            "time_signature": time_signature
        }
        notes.append(note)
    return notes


if __name__ == "__main__":
    score = music21.corpus.parse('bwv66.6')
    staves = featurizer.get_staff_indices(score)
    data = featurizer.load_data(score[staves[0]])
    tokens = featurizer.make_one_hot_features(data, False)
    new_data = detokenize(tokens)

    # Check the two datasets to verify that tokenizing/detokenizing is working ok
    compare_data(data, new_data)

    score1 = featurizer.unload_data(new_data)
    # score1.show()
