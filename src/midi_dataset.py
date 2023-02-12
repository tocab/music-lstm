"""
This module contains the definitions of the datasets.
"""

import glob
import itertools
import random
import multiprocessing as mp
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple

import numpy as np
from torch.utils.data import IterableDataset
from src.extract_notes import midi_to_notes

# Used for initial preprocessing
N_CPUS = 8


def process_data(input_data: Tuple[str, int, int]) -> List[Dict[str, np.ndarray]]:
    """
    Process input file by creating a sliding window of length sequence_length for the input notes and the output note.

    :param input_data: Tuple of (filename, sequence_length)

    :return: Preprocessed data, divided in input data and output data
    """
    filename = input_data[0]
    input_sequence_length = input_data[1]
    output_sequence_length = input_data[2]
    notes = midi_to_notes(filename)
    indices = [
        (start_index, start_index + input_sequence_length, start_index + input_sequence_length + output_sequence_length)
        for start_index in range(len(notes["pitch"]) - input_sequence_length - output_sequence_length)
    ]

    return [
        {
            "input_pitch": notes["pitch"][input_start_index: input_end_index],
            "input_step": notes["step"][input_start_index: input_end_index],
            "input_duration": notes["duration"][input_start_index: input_end_index],
            "output_pitch": notes["pitch"][input_end_index: output_end_index],
            "output_step": notes["step"][input_end_index: output_end_index],
            "output_duration": notes["duration"][input_end_index: output_end_index],
        }
        for input_start_index, input_end_index, output_end_index in indices
    ]


def process_data_seq2seq(input_data: Tuple[str, int, int]) -> List[Dict[str, np.ndarray]]:
    """
    Process input file by creating a sliding window of length sequence_length for the input notes and the output note.

    :param input_data: Tuple of (filename, sequence_length)

    :return: Preprocessed data, divided in encoder input, decoder input and decoder output
    """
    filename = input_data[0]
    input_sequence_length = input_data[1]
    output_sequence_length = input_data[2]
    notes = midi_to_notes(filename)
    indices = [
        (start_index, start_index + input_sequence_length, start_index + input_sequence_length + output_sequence_length)
        for start_index in range(len(notes["pitch"]) - input_sequence_length - output_sequence_length)
    ]

    return [
        {
            "encoder_input_pitch": notes["pitch"][input_start_index: input_end_index],
            "encoder_input_step": notes["step"][input_start_index: input_end_index],
            "encoder_input_duration": notes["duration"][input_start_index: input_end_index],
            "decoder_input_pitch": np.concatenate([[0], notes["pitch"][input_end_index: output_end_index-1]]),
            "decoder_input_step": np.concatenate([[0], notes["step"][input_end_index: output_end_index-1]]),
            "decoder_input_duration": np.concatenate([[0], notes["duration"][input_end_index: output_end_index-1]]),
            "decoder_output_pitch": notes["pitch"][input_end_index: output_end_index],
            "decoder_output_step": notes["step"][input_end_index: output_end_index],
            "decoder_output_duration": notes["duration"][input_end_index: output_end_index],
        }
        for input_start_index, input_end_index, output_end_index in indices
    ]


class MidiDataset(IterableDataset):
    """
    Define midi dataset class from IterableDataset.
    """

    def __init__(
            self,
            filenames: List[str],
            input_sequence_length: int,
            output_sequence_length: int,
            data_processing_method: Callable[[Tuple[str, int, int]], List[Dict[str, np.ndarray]]]
    ):
        """
        Init function.

        :param filenames: Names of the input midi files.
        :param input_sequence_length: Sequence length for the input
        :param output_sequence_length: Sequence length for the output
        """
        super(MidiDataset).__init__()
        self.filenames = filenames
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.data_processing_method = data_processing_method

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
                            self.data_processing_method,
                            [
                                (filename, self.input_sequence_length, self.output_sequence_length)
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
