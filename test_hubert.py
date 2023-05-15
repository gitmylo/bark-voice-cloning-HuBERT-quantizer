import os

import numpy
import torch
import torchaudio

from hubert.customtokenizer import CustomTokenizer
from hubert.pre_kmeans_hubert import CustomHubert


def test_hubert(path: str, model: str = 'model/hubert/hubert_base_ls960.pt'):
    hubert_model = CustomHubert(checkpoint_path=model)
    customtokenizer = CustomTokenizer()

    wav, sr = torchaudio.load(os.path.join(path, 'prepared', '0_wav.wav'))
    original = numpy.load(os.path.join(path, 'prepared', '0_semantic.npy'))

    out = hubert_model.forward(wav, input_sample_hz=sr)
    out_tokenized = customtokenizer.get_token(out)

    # print(out.shape, out_tokenized.shape)
    print(original[:-1], out_tokenized)
