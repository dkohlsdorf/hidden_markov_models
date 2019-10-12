import numpy as np
import matplotlib.pyplot as plt
import librosa


def mfcc_from_file(filename, components = 64, fft_win = 1024):
  audio, rate = librosa.load(filename, sr=None)
  mfcc = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=components, n_fft=fft_win, hop_length=fft_win//2)
  return mfcc.T