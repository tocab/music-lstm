"""
This module contains the definitions of the datasets.
"""

import glob
import itertools
import random
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import IterableDataset
from extract_notes import midi_to_notes

# Used for initial preprocessing
N_CPUS = 8


def process_data(input_data: Tuple[str, int]) -> List[Dict[str, np.ndarray]]:
    """
    Process input file by creating a sliding window of length sequence_length for the input notes and the output note.

    :param input_data: Tuple of (filename, sequence_length)

    :return:
    """
    filename = input_data[0]
    sequence_length = input_data[1]
    notes = midi_to_notes(filename)
    indices = [
        (start_index, start_index + sequence_length)
        for start_index in range(len(notes["pitch"]) - sequence_length)
    ]

    return [
        {
            "input_pitch": notes["pitch"][start_index:end_index],
            "input_step": notes["step"][start_index:end_index],
            "input_duration": notes["duration"][start_index:end_index],
            "output_pitch": notes["pitch"][end_index],
            "output_step": notes["step"][end_index],
            "output_duration": notes["duration"][end_index],
        }
        for start_index, end_index in indices
    ]


class MidiDataset(IterableDataset):
    """
    Define midi dataset class from IterableDataset.
    """

    def __init__(self, filenames: List[str], sequence_length: int = 50):
        """
        Init function.

        :param filenames: Names of the input midi files.
        :param sequence_length: Sequence length for the input sequence
        """
        super(MidiDataset).__init__()
        self.filenames = filenames
        self.sequence_length = sequence_length

        self.data = []

    def preprocess_data(self):
        """
        Executes data preprocessing to fill up self.data
        """
        with mp.Pool(N_CPUS) as p:
            self.data = list(
                itertools.chain.from_iterable(
                    tqdm(
                        p.imap(
                            process_data,
                            [
                                (filename, self.sequence_length)
                                for filename in self.filenames
                            ],
                        ),
                        total=len(self.filenames),
                    )
                )
            )

    def shuffle_data(self):
        """
        Shuffles data.
        """
        random.shuffle(self.data)
        print("Shuffling done.")

    def sample_data(self, sample_fraction: float):
        """
        Sample data using a sample fraction. The samples data will be removed from the dataset.
        This method can be used to create a validation set.

        :param sample_fraction: Rate of the data that should be sampled

        :return: Sampled data
        """
        sample_size = int(len(self.data) * sample_fraction)
        return [self.data.pop() for _ in range(sample_size)]

    def get_data(self):
        """
        Defines output generator that yields data from self.data
        """
        for data in self.data:
            yield data

    def __iter__(self):
        """
        Returns output generator.
        """
        return self.get_data()


if __name__ == "__main__":
    filenames = glob.glob("data/maestro-v3.0.0/**/*.mid*")
    ds = MidiDataset(filenames)
    ds.preprocess_data()
    ds.shuffle_data()
    validation_data = ds.sample_data(0.2)
    print(len(validation_data))
