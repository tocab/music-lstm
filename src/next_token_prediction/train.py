"""
Module to run the training.
"""

import datetime
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.midi_dataset import MidiDataset, process_data
from src.next_token_prediction.model import LSTMModel

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
BATCH_SIZE = 512


def step(
    batch: torch.Tensor,
    model: torch.nn.Module,
    classification_loss: torch.nn.CrossEntropyLoss,
    regression_loss: torch.nn.MSELoss,
) -> torch.Tensor:
    """
    Execute one training/validation step.

    :param batch: Containing input data and output data
    :param model: Model to use for prediction
    :param classification_loss: Loss for the classification of the pitches
    :param regression_loss: Loss for the regression of step and duration

    :return: Final loss that is the sum of the individual losses for pitch, step and duration
    """
    input_pitch = batch["input_pitch"].to(DEVICE)
    input_step = batch["input_step"].to(DEVICE)
    input_duration = batch["input_duration"].to(DEVICE)
    out = model(input_pitch, input_step, input_duration)

    if batch["output_pitch"].size()[1] == 1:
        batch["output_pitch"] = torch.squeeze(batch["output_pitch"])
        batch["output_step"] = torch.squeeze(batch["output_step"])
        batch["output_duration"] = torch.squeeze(batch["output_duration"])

    pitch_error = classification_loss(
        out[0], batch["output_pitch"].type(torch.LongTensor).to(DEVICE)
    )
    step_error = regression_loss(out[1], batch["output_step"].double().to(DEVICE))
    duration_error = regression_loss(
        out[2], batch["output_duration"].double().to(DEVICE)
    )
    final_error = pitch_error + step_error + duration_error
    return final_error


def train():
    """
    Execute the training.
    """
    # training config
    input_sequence_length = 10
    output_sequence_length = 1
    model = LSTMModel
    validation_file_ratio = 0.1

    training_start = datetime.datetime.now().isoformat()
    model_save_path = f"saved_models/{training_start}"
    os.makedirs(model_save_path)
    # Create dataset
    filenames = glob.glob("data/maestro-v3.0.0/**/*.mid*")
    random.shuffle(filenames)

    num_training_files = int(len(filenames) * (1-validation_file_ratio))
    print('Creating training set.')
    training_dataset = MidiDataset(
        filenames[:num_training_files],
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        data_processing_method=process_data
    )
    training_dataset.preprocess_data()
    training_dataset.shuffle_data()

    print('Creating training set.')
    validation_dataset = MidiDataset(
        filenames[num_training_files:],
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        data_processing_method=process_data
    )
    validation_dataset.preprocess_data()

    # Create data loaders
    training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE)
    validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    # losses
    class_loss = torch.nn.CrossEntropyLoss()
    regression_loss = torch.nn.MSELoss()

    # Create model
    lstm_model = model().to(DEVICE).double()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    summary_writer = SummaryWriter()

    training_steps = 0
    for epoch in range(100):
        print(f"Epoch {epoch}: Training")
        for batch in tqdm(training_data_loader, total=int(len(training_dataset.data) / BATCH_SIZE)):
            final_error = step(batch, lstm_model, class_loss, regression_loss)

            # Backpropagation
            optimizer.zero_grad()
            final_error.backward()
            optimizer.step()

            # Write to tensorboard
            summary_writer.add_scalar("Loss/train", final_error.item(), training_steps)
            training_steps += 1

        print(f"Epoch {epoch}: Validation")
        validation_error = []
        for batch in tqdm(
            validation_data_loader, total=int(len(validation_dataset.data) / BATCH_SIZE)
        ):
            final_error = step(batch, lstm_model, class_loss, regression_loss)
            validation_error.append(final_error.item())

        # Write to tensorboard
        summary_writer.add_scalar("Loss/validation", np.mean(validation_error), epoch)

        torch.save(lstm_model, f"{model_save_path}/model_epoch_{epoch}.pth")


if __name__ == "__main__":
    train()
