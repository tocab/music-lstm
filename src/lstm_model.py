"""
Module for the definition of the lstm model(s).
"""

from typing import Tuple

import torch
from torch import nn


# Define model
class LSTMModel(nn.Module):
    """
    Defines LSTM model from torch.nn.Module.
    """

    def __init__(
        self,
        embedding_size=64,
        lstm_hidden_size=512,
        num_pitches=128,
        bidirectional=True,
    ):
        """
        Init function.

        :param embedding_size: Size of the embedding for the pitches
        :param lstm_hidden_size: Size of the lstm weights
        :param num_pitches: Count of pitches that are contained in the data
        :param bidirectional: Whether to use a bidirectional lstm
        """
        super().__init__()
        self.pitch_embedding = nn.Embedding(num_pitches, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size + 2,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_layer_multiplier = 1
        if bidirectional:
            output_layer_multiplier = 2

        self.pitch_output_layer = nn.Linear(
            lstm_hidden_size * output_layer_multiplier, num_pitches
        )
        self.step_output_layer = nn.Linear(
            lstm_hidden_size * output_layer_multiplier, 1
        )
        self.duration_output_layer = nn.Linear(
            lstm_hidden_size * output_layer_multiplier, 1
        )

    def forward(
        self,
        input_pitch: torch.Tensor,
        input_step: torch.Tensor,
        input_duration: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward step of the model.

        :param input_pitch: Tensor with size (batch_size, time_steps)
        :param input_step: Tensor with size (batch_size, time_steps)
        :param input_duration: Tensor with size (batch_size, time_steps)

        :return: Prediction for next note, containing pitch, step and duration
        """
        # get inputs
        input_step = input_step.double()
        input_duration = input_duration.double()
        pitch_embedded = self.pitch_embedding(input_pitch)
        input_step = torch.unsqueeze(input_step, -1)
        input_duration = torch.unsqueeze(input_duration, -1)
        all_features = torch.cat((pitch_embedded, input_step, input_duration), dim=-1)
        lstm_output, _ = self.lstm(all_features)
        max_pooled, _ = torch.max(lstm_output, 1)

        pitch_output = self.pitch_output_layer(max_pooled)
        step_output = torch.squeeze(self.step_output_layer(max_pooled))
        duration_output = torch.squeeze(self.duration_output_layer(max_pooled))

        return pitch_output, step_output, duration_output
