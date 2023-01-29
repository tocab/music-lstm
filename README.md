# music-lstm
Make music with LSTMs

This project aims to reproduce and improve the results from the tensorflow tutorial [Generate music with an RNN](https://www.tensorflow.org/tutorials/audio/music_generation).
For that, the newer dataset maestro v3.0.0 is used and the implementation of the model is done with pytorch.

To run the project, it is recommended to use poetry to get all matching versions of the python packages.

### Download dataset
To download the dataset, run `python src/get_data.py`. This will download the maestro v3.0.0
dataset which contains midi files of classical compositions.

### Run training
The training can be started with `python src/train.py`. It also saves tensorboard logs, so for monitoring 
the training tensorboard can be used.

### Compose song
Once the model has finished the training, it can be used to generate a song with `python src/compose.py`.

## Iterations

### Iteration 1
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

For a better experience, an encoder-decoder model will be tried out in the next iteration.