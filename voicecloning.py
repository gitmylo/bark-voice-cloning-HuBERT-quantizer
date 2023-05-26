"""
git clone https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer.git
pip install huggingface_hub
cd /content/bark-voice-cloning-HuBERT-quantizer
pip install torchaudio torch encodec numpy fairseq audiolm_pytorch tensorboardX funcy
"""
from hubert.hubert_manager import make_sure_hubert_installed, make_sure_tokenizer_installed

HuBERTManager.make_sure_hubert_installed()
print("hubert_installed")
HuBERTManager.make_sure_tokenizer_installed()
print("tokenizer_installed")

"""add your file(as a wav), rename it to "audio.wav", and make sure it is in /content (same directory with config and sample_data)

Its recommended to shorten the audio file to 15-20 seconds, and to use the end, not the beginning

# add your audio now
"""

from hubert.pre_kmeans_hubert import CustomHubert
import torchaudio

# Load the HuBERT model,
# checkpoint_path should work fine with data/models/hubert/hubert.pt for the default config
hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt')

# Run the model to extract semantic features from an audio file, where wav is your audio file
wav, sr = torchaudio.load('/content/audio.wav') # This is where you load your wav, with soundfile or torchaudio for example

if wav.shape[0] == 2:  # Stereo to mono if needed
    wav = wav.mean(0, keepdim=True)

semantic_vectors = hubert_model.forward(wav, input_sample_hz=sr)

import torch
from hubert.customtokenizer import CustomTokenizer

# Load the CustomTokenizer model from a checkpoint
# With default config, you can use the pretrained model from huggingface
# With the default setup from HuBERTManager, this will be in data/models/hubert/tokenizer.pth
tokenizer = CustomTokenizer.load_from_checkpoint('/content/bark-voice-cloning-HuBERT-quantizer/data/models/hubert/tokenizer.pth')  # Automatically uses the right layers

# Process the semantic vectors from the previous HuBERT run (This works in batches, so you can send the entire HuBERT output)
semantic_tokens = tokenizer.get_token(semantic_vectors)

# Congratulations! You now have semantic tokens which can be used inside of a speaker prompt file.

from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined bythe bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
model.set_target_bandwidth(6.0)

# Load and pre-process the audio waveform
wav, sr = torchaudio.load("/content/audio.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

import numpy

fine_prompt = codes

coarse_prompt = fine_prompt[:2, :]

semantics = semantic_tokens

numpy.savez(semantic_prompt=semantics, fine_prompt=fine_prompt, coarse_prompt=coarse_prompt, file="helloWorld.npz")

"""now that we have the voice cloned as an npz, we can make text to speech with it!"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!git clone https://github.com/suno-ai/bark.git
# %cd bark

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Yay! Working voice cloning!
"""
audio_array = generate_audio(text=text_prompt, history_prompt='/content/bark-voice-cloning-HuBERT-quantizer/helloWorld.npz')

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)