"""
Module to run the training.
"""

import datetime
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from midi_dataset import MidiDataset
from lstm_model import LSTMModel

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
BATCH_SIZE = 1024


def step(
    batch: torch.Tensor,
    model: torch.nn.Module,
    classification_loss: torch.nn.CrossEntropyLoss,
    regression_loss: torch.nn.MSELoss,
) -> torch.Tensor:
    """
    Execute one trainig step.

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

    pitch_error = classification_loss(
        out[0], batch["output_pitch"].type(torch.LongTensor).to(DEVICE)
    )
    step_error = regression_loss(out[1], batch["output_step"].double().to(DEVICE))
    duration_error = regression_loss(
        out[2], batch["output_duration"].double().to(DEVICE)
    )
    final_error = pitch_error + step_error + duration_error
    return final_error


def run_training():
    """
    Execute the training.
    """
    training_start = datetime.datetime.now().isoformat()
    model_save_path = f"saved_models/{training_start}"
    os.makedirs(model_save_path)
    # Create dataset
    filenames = glob.glob("data/maestro-v3.0.0/**/*.mid*")
    ds = MidiDataset(filenames)
    ds.preprocess_data()
    ds.shuffle_data()

    # Create validation dataset
    validation_data = ds.sample_data(0.2)
    validation_ds = MidiDataset(filenames)
    validation_ds.data = validation_data

    # Create data loaders
    training_data_loader = DataLoader(ds, batch_size=BATCH_SIZE)
    validation_data_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE)

    # losses
    class_loss = torch.nn.CrossEntropyLoss()
    regression_loss = torch.nn.MSELoss()

    # Create model
    lstm_model = LSTMModel().to(DEVICE).double()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    summary_writer = SummaryWriter()

    training_steps = 0
    for epoch in range(100):
        print(f"Epoch {epoch}: Training")
        for batch in tqdm(training_data_loader, total=int(len(ds.data) / BATCH_SIZE)):
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
            validation_data_loader, total=int(len(validation_ds.data) / BATCH_SIZE)
        ):
            final_error = step(batch, lstm_model, class_loss, regression_loss)
            validation_error.append(final_error.item())

        # Write to tensorboard
        summary_writer.add_scalar("Loss/validation", np.mean(validation_error), epoch)

        torch.save(lstm_model, f"{model_save_path}/model_epoch_{epoch}.pth")


if __name__ == "__main__":
    run_training()
