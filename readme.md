# Bark voice cloning

## Please read
This code works on python 3.10, i have not tested it on other versions. Some older versions will have issues.

## Voice cloning with bark in high quality?
It's possible now.

https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/assets/36931363/516375e2-d699-44fe-a928-cd0411982049



## How do I clone a voice?
[examples on huggingface model page](https://huggingface.co/GitMylo/bark-voice-cloning)

[or use my webui](https://github.com/gitmylo/audio-webui)

[or use my huggingface voice cloning space](https://huggingface.co/spaces/GitMylo/bark-voice-cloning)

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

## For developers: Implementing voice cloning in your bark projects
* Simply clone this repo
* Then install requirements `pip3.10 install -r requirements.txt`
* make sure you are using python 3.10
* Add your audio.wav
* Run voiceCloning.py
* now to generate text to speech, run barkEx.py

## How do I train it myself?
Simply run the training commands.

A simple way to create semantic data and wavs for training, is with my script: [bark-data-gen](https://github.com/gitmylo/bark-data-gen). But remember that the creation of the wavs will take around the same time if not longer than the creation of the semantics. This can take a while to generate because of that.

For example, if you have a dataset with zips containing audio files, one zip for semantics, and one for the wav files. Inside of a folder called "Literature"

You should run `process.py --path Literature --mode prepare` for extracting all the data to one directory

You should run `process.py --path Literature --mode prepare2` for creating HuBERT semantic vectors, ready for training

You should run `process.py --path Literature --mode train` for training

And when your model has trained enough, you can run `process.py --path Literature --mode test` to test the latest model.

## Disclaimer
I am not responsible for audio generated using semantics created by this model. Just don't use it for illegal purposes.
