"""
Module to extract notes from midi and vice versa.
"""

import collections
import numpy as np
import pandas as pd
import pretty_midi
from typing import Dict


def midi_to_notes(midi_file: str) -> Dict[str, np.ndarray]:
    """
    Extract notes from a midi file.

    :param midi_file: Name of the midi file

    :return: Dict containing the extracted notes
    """
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes["pitch"].append(note.pitch)
        notes["start"].append(start)
        notes["end"].append(end)
        notes["step"].append(start - prev_start)
        notes["duration"].append(end - start)
        prev_start = start

    return {name: np.array(value) for name, value in notes.items()}


def notes_to_midi(
    notes: pd.DataFrame,
    instrument_name: str,
    velocity: int = 100,
) -> pretty_midi.PrettyMIDI:
    """
    Convert notes to midi file.

    :param notes: otes as dataframe, containing the columns pitch, step, duration
    :param instrument_name: Name of the instrument
    :param velocity: note loudness
    :return:
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note["step"])
        end = float(start + note["duration"])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note["pitch"]),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    return pm
