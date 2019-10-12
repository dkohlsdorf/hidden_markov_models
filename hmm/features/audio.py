import numpy as np
import  tensorflow.keras as K

from sksound.sounds import Sound
from numpy.fft import fft
from scipy.fftpack import dct


def spectrogram_from_file(filename, win=256, step=128):
  '''
  Read audio and convert to z-normalized spectrogram  

  filename: path to the file
  max_len: clip files
  '''
  sound = Sound(filename) 
  data  = sound.data
  if len(data.shape) > 1:
    data = data[:, 0]    
  spec = np.abs(spectrogram(data, win, step))[:, win//2:win]
  return spec 


def spectrogram(audio, win=256, step=128):
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


def cepstrum(spectrogram):
    '''
    Compute the cepstrum from a spectrogram
    '''
    x   = np.array([dct(np.log(np.square(frame))) for frame in spectrogram])
    mu  = np.mean(x, axis = 0)
    std = np.std(x,  axis = 0)
    return  (x - mu) / std
