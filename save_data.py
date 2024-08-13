"""
File: save_data.py

This module loads a music21 corpus, processes it, and saves it to a JSON file in
preparation for running on a HPC system.
"""

from pathlib import Path
import music21
import music21bindings
import json
import corpus

if __name__ == "__main__":
    print("Loading dataset...")
    # Get the corpus and prepare it as a dataset
    # scores = music_finder.prepare_directory(PATH, device)
    scores = corpus.get_m21_corpus('bach')

    # essen folksongs
    """
    opus = [list(op.scores) for op in [
        music21.corpus.parse("essenFolksong/altdeu10.abc"), music21.corpus.parse("essenFolksong/altdeu20.abc"),
        music21.corpus.parse("essenFolksong/ballad10.abc"), music21.corpus.parse("essenFolksong/ballad20.abc"),
        music21.corpus.parse("essenFolksong/ballad30.abc"), music21.corpus.parse("essenFolksong/ballad40.abc"),
        music21.corpus.parse("essenFolksong/ballad50.abc"), music21.corpus.parse("essenFolksong/ballad60.abc"),
        music21.corpus.parse("essenFolksong/ballad70.abc"), music21.corpus.parse("essenFolksong/ballad80.abc"),
        music21.corpus.parse("essenFolksong/boehme10.abc"), music21.corpus.parse("essenFolksong/boehme20.abc")
    ]]
    scores = list(itertools.chain.from_iterable(opus))
    """

    # Process the scores
    processed_score_list = []
    for score in scores:
        # Go through each staff in each score, and generate individual
        # sequences and labels for that staff
        for i in music21bindings.get_staff_indices(score):
            processed_score_list.append(music21bindings.load_data(score[i]))

    # Prepare the score for writing to JSON
    for score in processed_score_list:
        for note in score:
            note["quarterLength"] = str(note["quarterLength"])
            note["beat"] = str(note["beat"])

    # Output to JSON
    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/corpus1.json", "w") as output_json:
        output_json.write(json.dumps(processed_score_list))
    
