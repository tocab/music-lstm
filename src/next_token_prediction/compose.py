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

from src.extract_notes import notes_to_midi
from src.midi_dataset import MidiDataset, process_data
from torch.utils.data import DataLoader

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
INPUT_SEQUENCE_LENGTH = 10
OUTPUT_SEQUENCE_LENGTH = 1
SONG_LENGTH = 1000
INSTRUMENT_NAME = "Acoustic Grand Piano"
SAMPLING_RATE = 16000.0
MODEL = "saved_models/2023-02-12T10:23:33.782217/model_epoch_1.pth"

filenames = glob.glob("data/maestro-v3.0.0/**/*.mid*")
ds = MidiDataset(
    filenames[:2],
    input_sequence_length=INPUT_SEQUENCE_LENGTH,
    output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
    data_processing_method=process_data
)
ds.preprocess_data()
ds.shuffle_data()
data_sampler = DataLoader(ds, batch_size=1)
softmax = torch.nn.Softmax(dim=-1)

samples = [sample for sample in data_sampler]
sample = samples[0]

# Create empty array that get filled up in the loop
input_pitch = torch.zeros((1, SONG_LENGTH), dtype=sample["input_pitch"].dtype).to(
    DEVICE
)
input_step = torch.zeros((1, SONG_LENGTH), dtype=sample["input_step"].dtype).to(DEVICE)
input_duration = torch.zeros((1, SONG_LENGTH), dtype=sample["input_duration"].dtype).to(
    DEVICE
)

# Fill up first 50 steps
input_pitch[0, :INPUT_SEQUENCE_LENGTH] = sample["input_pitch"].to(DEVICE)
input_step[0, :INPUT_SEQUENCE_LENGTH] = sample["input_step"].to(DEVICE)
input_duration[0, :INPUT_SEQUENCE_LENGTH] = sample["input_duration"].to(DEVICE)

model = (
    torch.load(MODEL)
    .to(DEVICE)
    .double()
)

for step in tqdm(range(SONG_LENGTH - INPUT_SEQUENCE_LENGTH)):
    prediction = model(
        input_pitch[:, step : step + INPUT_SEQUENCE_LENGTH],
        input_step[:, step : step + INPUT_SEQUENCE_LENGTH],
        input_duration[:, step : step + INPUT_SEQUENCE_LENGTH],
    )
    input_pitch[0, step + INPUT_SEQUENCE_LENGTH] = torch.argmax(softmax(prediction[0]))
    input_step[0, step + INPUT_SEQUENCE_LENGTH] = prediction[1]
    input_duration[0, step + INPUT_SEQUENCE_LENGTH] = prediction[2]

output_df = pd.DataFrame(
    {
        "pitch": input_pitch.cpu().detach().flatten(),
        "step": input_step.cpu().detach().flatten(),
        "duration": input_duration.cpu().detach().flatten(),
    }
)

midi_song = notes_to_midi(output_df, INSTRUMENT_NAME)
song_name = datetime.datetime.now().isoformat()

if not os.path.exists('songs'):
    os.makedirs('songs')
midi_song.write(f"songs/{song_name}.mid")
waveform = midi_song.fluidsynth(fs=SAMPLING_RATE)
sd.play(waveform, samplerate=SAMPLING_RATE, blocking=True)
