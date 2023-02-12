"""
Module to compose a song from a trained model.
"""

import datetime
import glob
import os
import pandas as pd
import torch
import sounddevice as sd
from tqdm import tqdm
from typing import Tuple

from src.extract_notes import notes_to_midi
from src.midi_dataset import MidiDataset, process_data_seq2seq
from torch.utils.data import DataLoader

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# DEVICE = "cpu"
INPUT_SEQUENCE_LENGTH = 10
OUTPUT_SEQUENCE_LENGTH = 50
SONG_LENGTH = 600
INSTRUMENT_NAME = "Acoustic Grand Piano"
SAMPLING_RATE = 16000.0
MODEL = "saved_models/2023-02-07T08:17:07.282464/model_epoch_0.pth"
BEAM_WIDTH = 20
COUNT_PITCHES = 128

filenames = glob.glob("data/maestro-v3.0.0/**/*.mid*")

ds = MidiDataset(
    filenames[:2],
    input_sequence_length=INPUT_SEQUENCE_LENGTH,
    output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
    data_processing_method=process_data_seq2seq
)
ds.preprocess_data()
ds.shuffle_data()
data_sampler = DataLoader(ds, batch_size=1)
loop_softmax = torch.nn.Softmax(dim=-1)
final_softmax = torch.nn.Softmax(dim=0)
test_softmax = torch.nn.Softmax(dim=1)

samples = [sample for sample in data_sampler]
sample = samples[0]

# Encoder: Create empty array that get filled up in the loop
encoder_input_pitch = torch.zeros((BEAM_WIDTH, SONG_LENGTH), dtype=sample["encoder_input_pitch"].dtype)
encoder_input_step = torch.zeros((BEAM_WIDTH, SONG_LENGTH), dtype=sample["encoder_input_step"].dtype)
encoder_input_duration = torch.zeros((BEAM_WIDTH, SONG_LENGTH), dtype=sample["encoder_input_duration"].dtype)

# Fill up first steps
for beam_step in range(BEAM_WIDTH):
    encoder_input_pitch[beam_step, :INPUT_SEQUENCE_LENGTH] = sample["encoder_input_pitch"]
    encoder_input_step[beam_step, :INPUT_SEQUENCE_LENGTH] = sample["encoder_input_step"]
    encoder_input_duration[beam_step, :INPUT_SEQUENCE_LENGTH] = sample["encoder_input_duration"]

encoder_input_pitch = encoder_input_pitch.to(DEVICE)
encoder_input_step = encoder_input_step.to(DEVICE)
encoder_input_duration = encoder_input_duration.to(DEVICE)

model = (
    torch.load(MODEL)
    .to(DEVICE)
    .double()
    .eval()
)

# Decoder: Create empty array that get filled up in the loop
decoder_input_pitch = torch.zeros((BEAM_WIDTH, OUTPUT_SEQUENCE_LENGTH + 1),dtype=sample["encoder_input_pitch"].dtype).to(DEVICE)
decoder_input_step = torch.zeros((BEAM_WIDTH, OUTPUT_SEQUENCE_LENGTH + 1), dtype=sample["encoder_input_step"].dtype).to(DEVICE)
decoder_input_duration = torch.zeros((BEAM_WIDTH, OUTPUT_SEQUENCE_LENGTH + 1), dtype=sample["encoder_input_duration"].dtype).to(DEVICE)

# beam errors
beam_errors = torch.zeros((BEAM_WIDTH,)).to(DEVICE)


def top_k(input_tensor: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return top k elements and corresponding indices in input tensor.

    :param input_tensor: Input tensor (1d array)
    :param k: Number of top elements to return

    :return: Tuple containing top values and corresponding indices in input_tensor
    """
    indices = []
    values = []
    for _ in range(k):
        max_value, max_index = torch.max(input_tensor, -1)
        values.append(max_value)
        indices.append(max_index)
        input_tensor[max_index] = -float('inf')
    return torch.stack(values), torch.stack(indices)


for step in tqdm(range(INPUT_SEQUENCE_LENGTH, SONG_LENGTH, OUTPUT_SEQUENCE_LENGTH)):
    # Predictions for every time step. start with empty array that only contains the start token to receive the
    # first output token. Put the first output token into the second position (after the start token) and let
    # model predict again. Repeat it for all output time steps.
    encoder_input_start_index = step-INPUT_SEQUENCE_LENGTH
    encoder_input_end_index = step

    beam_errors = beam_errors.zero_()
    for decoder_step in range(OUTPUT_SEQUENCE_LENGTH):
        temp_prediction = model(
            encoder_input_pitch[:, encoder_input_start_index: encoder_input_end_index],
            encoder_input_step[:, encoder_input_start_index: encoder_input_end_index],
            encoder_input_duration[:, encoder_input_start_index: encoder_input_end_index],
            decoder_input_pitch,
            decoder_input_step,
            decoder_input_duration,
            OUTPUT_SEQUENCE_LENGTH
        )

        if decoder_step == 0:
            log_probs = torch.log(loop_softmax(temp_prediction[0][0, :, decoder_step]))
            top_values, top_indices = top_k(log_probs, BEAM_WIDTH)

            decoder_input_pitch[:, decoder_step + 1] = top_indices
            beam_errors += top_values
        else:
            log_probs = torch.log(loop_softmax(temp_prediction[0][:, :, decoder_step]))

            for beam_step in range(BEAM_WIDTH):
                log_probs[beam_step, :] += beam_errors[beam_step]

            top_values, top_indices = top_k(torch.flatten(log_probs), BEAM_WIDTH)

            indices_before = top_indices // COUNT_PITCHES
            indices_new = top_indices % COUNT_PITCHES

            beam_errors = top_values

            decoder_input_pitch = decoder_input_pitch[indices_before]
            decoder_input_pitch[:, decoder_step + 1] = indices_new

        if BEAM_WIDTH == 1:
            temp_prediction = list(temp_prediction)
            temp_prediction[1] = torch.unsqueeze(temp_prediction[1], dim=0)
            temp_prediction[2] = torch.unsqueeze(temp_prediction[2], dim=0)

        decoder_input_step[:, decoder_step + 1] = temp_prediction[1][:, decoder_step]
        decoder_input_duration[:, decoder_step + 1] = temp_prediction[2][:, decoder_step]

    max_log_prob = torch.argmax(beam_errors)

    # determine how many time steps should be appended to the encoder array. In case the last loop round
    # contains lesser time steps than the others, change size_to_append.
    size_to_append = OUTPUT_SEQUENCE_LENGTH
    if SONG_LENGTH - step < OUTPUT_SEQUENCE_LENGTH:
        size_to_append = SONG_LENGTH - step

    # Append decoder output to encoder input
    for beam_step in range(BEAM_WIDTH):
        encoder_input_pitch[beam_step, step: step + size_to_append] = decoder_input_pitch[max_log_prob, 1:size_to_append+1]
        encoder_input_step[beam_step, step: step + size_to_append] = decoder_input_step[max_log_prob, 1:size_to_append+1]
        encoder_input_duration[beam_step, step: step + size_to_append] = decoder_input_duration[max_log_prob, 1:size_to_append+1]

output_df = pd.DataFrame(
    {
        "pitch": encoder_input_pitch[0, :].cpu().detach().flatten(),
        "step": encoder_input_step[0, :].cpu().detach().flatten(),
        "duration": encoder_input_duration[0, :].cpu().detach().flatten(),
    }
)

midi_song = notes_to_midi(output_df, INSTRUMENT_NAME)
song_name = datetime.datetime.now().isoformat()

if not os.path.exists('songs'):
    os.makedirs('songs')
midi_song.write(f"songs/{song_name}.mid")
waveform = midi_song.fluidsynth(fs=SAMPLING_RATE)
sd.play(waveform, samplerate=SAMPLING_RATE, blocking=True)
