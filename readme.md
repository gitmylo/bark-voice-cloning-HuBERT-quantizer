# Bark voice cloning

## Voice cloning with bark at high quality?
It's possible now.
[joe biden example](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/raw/master/examples/biden_example.mov) (Idk how to embed videos lol)

## How do I clone a voice?
[examples on huggingface model page](https://huggingface.co/GitMylo/bark-voice-cloning)

## Voices cloned aren't very convincing, why are other people's cloned voices better than mine?
Make sure these things are **NOT** in your voice input: (in no particular order)
* Noise (You can use a noise remover before)
* Music (There are also music remover tools) (Unless you want music in the background)
* A cut-off at the end (This will cause it to try and continue on the generation)
* Under 1 second of training data (i personally suggest around 10 seconds for good potential, but i've had great results with 5 seconds as well.)

What makes for good prompt audio? (in no particular order)
* Clearly spoken
* No weird background noises
* Only one speaker
* Audio which ends after a sentence ends
* Regular/common voice (They usually have more success, it's still capable of cloning complex voices, but not as good at it)
* Around 10 seconds of data

## How do I train it myself?
Simply run the training commands.
For example, if you have a dataset with zips containing audio files, one zip for semantics, and one for the wav files. Inside of a folder called "Literature"

You should run `process.py --path Literature --mode prepare` for extracting all the data to one directory

You should run `process.py --path Literature --mode prepare2` for creating HuBERT semantic vectors, ready for training

You should run `process.py --path Literature --mode train` for training

And when your model has trained enough, you can run `process.py --path Literature --mode test` to test the latest model.

## Disclaimer
I am not responsible for audio generated using semantics created by this model. Just don't use it for illegal purposes.
