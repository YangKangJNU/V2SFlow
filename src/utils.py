import math
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

import torch

from src.stft import TacotronSTFT
from scipy.io.wavfile import read



def default(v, d):
    return v if exists(v) else d

def exists(v):
    return v is not None

def lens_to_mask(t, length = None) -> bool:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

def plot_f0(y1, y2):
    x = np.linspace(0, 1, y1.shape[1])
    fig, ax = plt.subplots()
    ax.plot(x, y1[0], label='predict_f0_norm')
    ax.plot(x, y2[0], label='f0_norm')
    ax.legend()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class STFT():
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, filter_length, hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)

    def get_mel(self, audio):
        audio = audio.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)
        return melspec
    
def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

def normalise_mel(melspec, min_val=math.log(1e-5)): 
    melspec = ((melspec - min_val) / (-min_val / 2)) - 1    #log(1e-5)~2 --> -1~1
    return melspec

def denormalise_mel(melspec, min_val=math.log(1e-5)): 
    melspec = ((melspec + 1) * (-min_val / 2)) + min_val
    return melspec

def to_numpy(t):
    return t.detach().cpu().numpy()

def plot_spectrogram(spectrogram):
    spectrogram = to_numpy(spectrogram)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram.T, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig

# SMN-logf0 from PFlow-VC
def SMN_logF0(f0):
    ii = f0 == 0
    logf0 = np.log(f0 + 1e-5)
    E = np.sum(logf0[~ii]) / np.sum(~ii)
    smn_logf0 = logf0 - E
    smn_logf0[ii] = 0
    return smn_logf0
    