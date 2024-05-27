"""
File: tests.py

This module has tests to verify that feature encoding works properly.
"""

from fractions import Fraction
import music_features
import music_featurizer
import music21

def detokenize(tokens):
    """
    Detokenizes one-hot features and returns them to regular encoding
    :param tokens: A tensor of one-hot encoded instances
    :return: The regular encoding
    """
    i = (
        (0, len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING)), 
        (len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING), len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING)),
        (len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING), len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING)),
        (len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING), len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING)),
        (len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING), len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING) + len(music_features.MELODIC_INTERVAL_ENCODING)),
        (len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING) + len(music_features.MELODIC_INTERVAL_ENCODING), len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING) + len(music_features.MELODIC_INTERVAL_ENCODING) + len(music_features.KEY_SIGNATURE_ENCODING)),
        (len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING) + len(music_features.MELODIC_INTERVAL_ENCODING) + len(music_features.KEY_SIGNATURE_ENCODING), len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING) + len(music_features.MELODIC_INTERVAL_ENCODING) + len(music_features.KEY_SIGNATURE_ENCODING) + len(music_features.MODE_ENCODING)),
        (len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING) + len(music_features.MELODIC_INTERVAL_ENCODING) + len(music_features.KEY_SIGNATURE_ENCODING) + len(music_features.MODE_ENCODING), len(music_features.LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(music_features.QUARTER_LENGTH_ENCODING) + len(music_features.BEAT_ENCODING) + len(music_features.PITCH_CLASS_ENCODING) + len(music_features.MELODIC_INTERVAL_ENCODING) + len(music_features.KEY_SIGNATURE_ENCODING) + len(music_features.MODE_ENCODING) + len(music_features.TIME_SIGNATURE_ENCODING)),
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
        letter_accidental_octave = music_features.REVERSE_LETTER_ACCIDENTAL_OCTAVE_ENCODING[letter_accidental_octave.argmax().item()]
        quarter_length = music_features.REVERSE_QUARTER_LENGTH_ENCODING[quarter_length.argmax().item()]
        beat = music_features.REVERSE_BEAT_ENCODING[beat.argmax().item()]
        pitch_class = music_features.REVERSE_PITCH_CLASS_ENCODING[pitch_class.argmax().item()]
        melodic_interval = music_features.REVERSE_MELODIC_INTERVAL_ENCODING[melodic_interval.argmax().item()]
        key_signature = music_features.REVERSE_KEY_SIGNATURE_ENCODING[key_signature.argmax().item()]
        mode = music_features.REVERSE_MODE_ENCODING[mode.argmax().item()]
        time_signature = music_features.REVERSE_TIME_SIGNATURE_ENCODING[time_signature.argmax().item()]
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


score = music21.corpus.parse('bwv66.6')
staves = music_featurizer.get_staff_indices(score)
data = music_featurizer.load_data(score[staves[0]])
tokens = music_featurizer.make_one_hot_features(data, False)
new_data = detokenize(tokens)
score1 = music_featurizer.unload_data(new_data)
score1.show()
