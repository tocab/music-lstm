# music-lstm
Make music with LSTMs

This project aims to reproduce and improve the results from the tensorflow tutorial [Generate music with an RNN](https://www.tensorflow.org/tutorials/audio/music_generation).
For that, the newer dataset maestro v3.0.0 is used and the implementation of the model is done with pytorch.

To run the project, it is recommended to use poetry to get all matching versions of the python packages.

### Download dataset
To download the dataset, run `python src/get_data.py`. This will download the maestro v3.0.0
dataset which contains midi files of classical compositions.

### Run training
The training can be started with `python src/{MODEL_TYPE}/train.py`. It also saves tensorboard logs, so for monitoring 
the training tensorboard can be used.

### Compose song
Once the model has finished the training, it can be used to generate a song with `python src/{MODEL_TYPE}/compose.py`.

## Iterations

### Iteration 1: Next token prediction
The first iteration aimed to reproduce the results from the tensorflow tutorial that uses a simple
LSTM that receives an input sequence and predictions the next note. To generate a whole song, this
process is repeated for a pre-defined amount of steps. In this first iteration, the LSTM is trained
to use the first 50 notes as input to predict the next note.

The results after running the training for 8 epochs show that this way to train the LSTM is not
good enough to produce good results. The main procedural generation of a song. While for the first
generation step a random part of an original song is used as the input, the following inputs contain
more data that has been generated from the LSTM until the whole input sequence is created by it.
This lead to an unnatural melody with a lot of repetitions. In my cases, the last 500 notes of a 
song only switch between 2 notes.

### Iteration 2: Sequence to sequence
In the second iteration, a sequence to sequence approach is tried out to generate music. For that,
first the input data needs to be changed to input for the encoder model, input for the decoder model
and output of the decoder model. For the encoder of the input, a bidirectional LSTM is used. The output
of that is passed to a projection layer, which created two arrays for the initial cell state of the
LSTM output decoder. The decoder then generates the next tokens for a pre-defined length.

For the hyper parameters, an encoder input length of 10 and decoder output length of 50 is set. Different
hidden sizes have been tried out in the training. It turns out that a bigger hidden weight size
helps to get better validation resutls, but also increases the training time a lot. A hidden size
of 1024 for the encoder in decoder gave the best results after one epoch of training but lead to
overfitting in the second epoch.

To generate songs, the first 10 notes of a known song from the data are used to generate the next 50
notes. After that, the generation of the song continues on the last 10 notes that have been generated so
that it is possible to create songs of variable length. To improve the song quality, also beam search
is introduced in the process of song creation. With that, the probability of the whole next 50 notes
sequence should be maximized instead of greedily taking only the next note with the highest probability.

Overall, the quality of the generated songs is better than in the next token prediction. Especially
the problem of being stuck in one or two notes is being solved by only generating 50 tokens by the
decoder and then creating a new encoding with the last 10 tokens by the encoder, which resets
repetitions in some cases.
