"""
File: feature_definitions.py

This module has feature maps for music featurization.
"""

# Letter name and accidental
LETTER_NAME_ENCODING = {"C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "A": 6, "B": 7}
ACCIDENTAL_NAME_ENCODING = {"None": 0, 'double-flat': 1, 'double-sharp': 2, 'flat': 3, 'half-flat': 4, 'half-sharp': 5, 
                             'natural': 6, 'one-and-a-half-flat': 7, 'one-and-a-half-sharp': 8, 'quadruple-flat': 9, 
                             'quadruple-sharp': 10, 'sharp': 11, 'triple-flat': 12, 'triple-sharp': 13}
REVERSE_LETTER_NAME_ENCODING = {0: "None", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "A", 7: "B"}
REVERSE_ACCIDENTAL_NAME_ENCODING = {0: "None", 1: 'double-flat', 2: 'double-sharp', 3: 'flat', 4: 'half-flat', 
                                     5: 'half-sharp', 6: 'natural', 7: 'one-and-a-half-flat', 8: 'one-and-a-half-sharp', 
                                     9: 'quadruple-flat', 10: 'quadruple-sharp', 11: 'sharp', 12: 'triple-flat', 13: 'triple-sharp'}
OCTAVE_ENCODING = {}
REVERSE_OCTAVE_ENCODING = {0: "None"}
for i in range(1, 14+1):
    OCTAVE_ENCODING[str(i-1)] = i
    REVERSE_OCTAVE_ENCODING[i] = str(i-1)

# Letter name plus accidental
LETTER_ACCIDENTAL_ENCODING = {"None|None": 0}
REVERSE_LETTER_ACCIDENTAL_ENCODING = {0: "None|None"}

i = 1
for key_letter in LETTER_NAME_ENCODING:
    for key_accidental in ACCIDENTAL_NAME_ENCODING:
        LETTER_ACCIDENTAL_ENCODING[f"{key_letter}|{key_accidental}"] = i
        REVERSE_LETTER_ACCIDENTAL_ENCODING[i] = f"{key_letter}|{key_accidental}"
        i += 1

# Letter name plus accidental plus octave
LETTER_ACCIDENTAL_OCTAVE_ENCODING = {"None|None|None": 0}
REVERSE_LETTER_ACCIDENTAL_OCTAVE_ENCODING = {0: "None|None|None"}

i = 1
for key_letter in LETTER_NAME_ENCODING:
    for key_accidental in ACCIDENTAL_NAME_ENCODING:
        for key_octave in OCTAVE_ENCODING:
            LETTER_ACCIDENTAL_OCTAVE_ENCODING[f"{key_letter}|{key_accidental}|{key_octave}"] = i
            REVERSE_LETTER_ACCIDENTAL_OCTAVE_ENCODING[i] = f"{key_letter}|{key_accidental}|{key_octave}"
            i += 1

LETTER_NAME_ENCODING["None"] = 0
OCTAVE_ENCODING["None"] = 0

# Other features
ACCIDENTAL_ENCODING = {"None": 0, "-2.0": 1, "-1.0": 2, "0.0": 3, "1.0": 4, "2.0": 5}
PITCH_CLASS_ENCODING = {"None": 0}
PITCH_SPACE_ENCODING = {"None": 0}
MELODIC_INTERVAL_ENCODING = {"None": 0}
KEY_SIGNATURE_ENCODING = {"None": 0}
MODE_ENCODING = {"None": 0, "major": 1, "minor": 2}
BEAT_ENCODING = {"None": 0}
QUARTER_LENGTH_ENCODING = {"None": 0}
REVERSE_PITCH_CLASS_ENCODING = {0: "None"}
REVERSE_PITCH_SPACE_ENCODING = {0: "None"}
REVERSE_MELODIC_INTERVAL_ENCODING = {0: "None"}
REVERSE_KEY_SIGNATURE_ENCODING = {0: "None"}
REVERSE_MODE_ENCODING = {0: "None", 1: "major", 2: "minor"}
REVERSE_BEAT_ENCODING = {0: "None"}
REVERSE_QUARTER_LENGTH_ENCODING = {0: "None"}

# Lets you convert accidental names to chromatic alteration
ACCIDENTAL_NAME_TO_ALTER_ENCODING = {"None": 0.0, 'double-flat': 2.0, 'double-sharp': -2.0, 'flat': 1.0, 'half-flat': 0.5, 'half-sharp': -0.5, 
                                     'natural': 0.0, 'one-and-a-half-flat': 1.5, 'one-and-a-half-sharp': -1.5, 'quadruple-flat': 4.0, 
                                     'quadruple-sharp': -4.0, 'sharp': -1.0, 'triple-flat': 3.0, 'triple-sharp': -3.0}

ACCIDENTAL_ALTER_TO_NAME_ENCODING = {0.0: "None", 2.0: 'double-flat', -2.0: 'double-sharp', 1.0: 'flat', 0.5: 'half-flat', -0.5: 'half-sharp', 
                                      1.5: 'one-and-a-half-flat', -1.5: 'one-and-a-half-sharp', 4.0: 'quadruple-flat', 
                                      -4.0: 'quadruple-sharp', -1.0: 'sharp', 3.0: 'triple-flat', -3.0: 'triple-sharp'}

##########################################################################
# Generate pitch encoding
##########################################################################

for i in range(1, 128+1):
    PITCH_SPACE_ENCODING[str(float(i-1))] = i 
    REVERSE_PITCH_SPACE_ENCODING[i] = str(float(i-1))

for i in range(1, 12+1):
    PITCH_CLASS_ENCODING[str(float(i-1))] = i
    REVERSE_PITCH_CLASS_ENCODING[i] = str(float(i-1))

for i in range(1, 128+1):
    PITCH_SPACE_ENCODING[str(float(i-1))] = i 
    REVERSE_PITCH_SPACE_ENCODING[i] = str(float(i-1))

for i in range(1, 23+1):
    MELODIC_INTERVAL_ENCODING[str(float(i-12))] = i 
    REVERSE_MELODIC_INTERVAL_ENCODING[i] = str(float(i-12))

for i in range(1, 15+1):
    KEY_SIGNATURE_ENCODING[str(i-8)] = i 
    REVERSE_KEY_SIGNATURE_ENCODING[i] = str(i-8)

##########################################################################
# Generate beat and quarter length encoding
##########################################################################

# This sets the maximum note duration in quarter notes that the model can handle.
_MAX_QUARTER_LENGTH = 8

idx_quarter_length = 1

# Quarters
for i in range(1, _MAX_QUARTER_LENGTH):
    QUARTER_LENGTH_ENCODING[f"{i}"] = idx_quarter_length
    REVERSE_QUARTER_LENGTH_ENCODING[idx_quarter_length] = f"{i}"
    BEAT_ENCODING[f"{i}"] = idx_quarter_length
    REVERSE_BEAT_ENCODING[idx_quarter_length] = f"{i}"
    idx_quarter_length += 1

# 8ths, triplet 8ths, 16ths, triplet 16ths, 32nds. The first value
# in the tuple is the quarter length denominator, and the second value
# is a step value. The step value helps to avoid duplicate duration
# values in the encoding. In general it should probably be 2, except 
# for triplet eighths.
for denominator, step in [(2, 2), (3, 1), (4, 2), (6, 2), (8, 2)]:
    for i in range(1, _MAX_QUARTER_LENGTH * denominator, step):
        # This condition catches duplicate durations for subdivisions of 3 and 6
        if denominator % 3 != 0 or i % 3 != 0:
            QUARTER_LENGTH_ENCODING[f"{i}/{denominator}"] = idx_quarter_length
            REVERSE_QUARTER_LENGTH_ENCODING[idx_quarter_length] = f"{i}/{denominator}"
            BEAT_ENCODING[f"{i}/{denominator}"] = idx_quarter_length
            REVERSE_BEAT_ENCODING[idx_quarter_length] = f"{i}/{denominator}"
            idx_quarter_length += 1

TIME_SIGNATURE_ENCODING = {"None": 0}
REVERSE_TIME_SIGNATURE_ENCODING = {0: "None"}

i = 1
for j in [1, 2, 3, 4, 6, 8, 9, 12]:
    for k in [1, 2, 4, 8]:
        TIME_SIGNATURE_ENCODING[f"{j}/{k}"] = i
        REVERSE_TIME_SIGNATURE_ENCODING[i] = f"{j}/{k}"
        i += 1

###################################################################################################################
# The total number of features and outputs for the model. This can change from time to time, and must be updated!
###################################################################################################################
NUM_FEATURES = len(LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(QUARTER_LENGTH_ENCODING) + len(BEAT_ENCODING) + \
               len(PITCH_CLASS_ENCODING) + len(MELODIC_INTERVAL_ENCODING) + len(KEY_SIGNATURE_ENCODING)  + \
               len(MODE_ENCODING) + len(TIME_SIGNATURE_ENCODING)
NUM_OUTPUTS = len(LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(QUARTER_LENGTH_ENCODING)
