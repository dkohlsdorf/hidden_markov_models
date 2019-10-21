import numpy as np
import matplotlib.pyplot as plt
import librosa
from numpy.fft import fft
from scipy.io import wavfile



def mfcc_from_file(filename, components = 16, fft_win = 512):
  audio, rate = librosa.load(filename, sr=None)
  mfcc = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=components, n_fft=fft_win, hop_length=fft_win//2)
  return mfcc.T

def fwd_spectrogram(audio, win=512, step=256):
    '''
    Compute the spectrogram of audio data

    audio: one channel audio
    win: window size for dft sliding window
    step: step size for dft sliding windo
    '''
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = fft(audio[i - win: i] * hanning)
        spectrogram.append(dft)
    return np.array(spectrogram)

def spectrogram_from_file(filename, win=512, step=256):
    '''
    Read audio and convert to z-normalized spectrogram  
    filename: path to the file
    max_len: clip files
    '''
    fs, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]    
    start = win // 2
    spec = np.abs(fwd_spectrogram(data))[:, start:win]
    return spec

