"""
Module for the definition of the lstm model(s).
"""

from typing import Tuple

import torch
from torch import nn


# Define model
class LSTMEncoderDecoder(nn.Module):
    """
    Defines LSTMEncoderDecoder model from torch.nn.Module.
    """

    def __init__(
        self,
        embedding_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        encoding_size: int,
        num_pitches: int,
        encoder_bidirectional: bool
    ):
        """
        Init function.

        :param embedding_size: Size of the embedding for the pitches
        :param encoder_hidden_size: Size of the hidden weights for the lstm encoder
        :param encoder_hidden_size: Size of the hidden weights for the lstm decoder
        :param encoding_size: Size of the encoded input sequence
        :param num_pitches: Count of pitches that are contained in the data
        :param encoder_bidirectional: Whether to use a bidirectional lstm for encoder
        """
        super().__init__()
        self.pitch_embedding = nn.Embedding(num_pitches, embedding_size)
        self.encoder = nn.LSTM(
            input_size=embedding_size + 2,
            hidden_size=encoder_hidden_size,
            batch_first=True,
            bidirectional=encoder_bidirectional,
        )
        output_layer_multiplier = 1
        if encoder_bidirectional:
            output_layer_multiplier = 2

        self.encoding_layer_h = nn.Linear(
            encoder_hidden_size * output_layer_multiplier, encoding_size
        )
        self.encoding_layer_c = nn.Linear(
            encoder_hidden_size * output_layer_multiplier, encoding_size
        )

        self.decoder = nn.LSTM(
            input_size=embedding_size + 2,
            hidden_size=decoder_hidden_size,
            batch_first=True
        )

        self.pitch_output_layer = nn.Linear(decoder_hidden_size, num_pitches)
        self.step_output_layer = nn.Linear(encoder_hidden_size, 1)
        self.duration_output_layer = nn.Linear(encoder_hidden_size, 1)
        self.dropout = nn.Dropout()

    def forward(
        self,
        encoder_input_pitch: torch.Tensor,
        encoder_input_step: torch.Tensor,
        encoder_input_duration: torch.Tensor,
        decoder_input_pitch: torch.Tensor,
        decoder_input_step: torch.Tensor,
        decoder_input_duration: torch.Tensor,
        sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward step of the model.

        :param encoder_input_pitch: Encoder pitch input: Tensor with size (batch_size, time_steps)
        :param encoder_input_step: Encoder step input: Tensor with size (batch_size, time_steps)
        :param encoder_input_duration: Encoder duration input: Tensor with size (batch_size, time_steps)
        :param decoder_input_pitch: Decoder pitch input: Tensor with size (batch_size, time_steps)
        :param decoder_input_step: Decoder step input: Tensor with size (batch_size, time_steps)
        :param decoder_input_duration: Decoder duration input: Tensor with size (batch_size, time_steps)
        :param sequence_length: Number of time steps

        :return: Prediction for next notes (pitches, steps, durations) for next sequence_length time steps
        """
        # process encoder inputs
        input_step = encoder_input_step.double()
        input_duration = encoder_input_duration.double()
        pitch_embedded = self.pitch_embedding(encoder_input_pitch)
        input_step = torch.unsqueeze(input_step, -1)
        input_duration = torch.unsqueeze(input_duration, -1)
        all_encoder_features = torch.cat((pitch_embedded, input_step, input_duration), dim=-1)

        # process decoder inputs
        input_step = decoder_input_step.double()
        input_duration = decoder_input_duration.double()
        pitch_embedded = self.pitch_embedding(decoder_input_pitch)
        input_step = torch.unsqueeze(input_step, -1)
        input_duration = torch.unsqueeze(input_duration, -1)
        all_decoder_features = torch.cat((pitch_embedded, input_step, input_duration), dim=-1)

        # Encoder
        encoder_output, enc = self.encoder(all_encoder_features)
        max_pooled, _ = torch.max(encoder_output, 1)
        encoding_h = torch.unsqueeze(self.encoding_layer_h(max_pooled), 0)
        encoding_c = torch.unsqueeze(self.encoding_layer_c(max_pooled), 0)

        # Dropout
        # Assumption: Noise to decoder input helps to generalize better on self-generated data.
        encoding_h = self.dropout(encoding_h)
        encoding_c = self.dropout(encoding_c)

        # Decoder
        decoder_output, _ = self.decoder(all_decoder_features, (encoding_h, encoding_c))

        pitch_outputs = []
        step_outputs = []
        duration_outputs = []
        for output_projection_step in range(sequence_length):
            pitch_outputs.append(self.pitch_output_layer(decoder_output[:, output_projection_step, :]))
            step_outputs.append(torch.squeeze(self.step_output_layer(decoder_output[:, output_projection_step, :])))
            duration_outputs.append(torch.squeeze(self.duration_output_layer(decoder_output[:, output_projection_step, :])))

        pitch_outputs = torch.stack(pitch_outputs, dim=-1)
        step_outputs = torch.stack(step_outputs, dim=-1)
        duration_outputs = torch.stack(duration_outputs, dim=-1)

        return pitch_outputs, step_outputs, duration_outputs
