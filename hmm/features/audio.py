import numpy as np

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


def spectrogram(audio, win=256, step=64):
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


def cepstrum(spectrogram, target = 16):
    '''
    Compute the cepstrum from a spectrogram
    '''
    _, d = spectrogram.shape
    window  = d // target
    step    = window // 2
    hanning = np.hanning(window)
    print(window, step)
    result  = [] 
    for frame in spectrogram:
        ceps = np.zeros(target) 
        for i in range(window, d, step):
            ceps[(i - 1) // window] = np.log(np.sum(frame[i - window : i] * hanning) + 1.0)
        result.append(dct(ceps))    
    result = np.array(result)
    mu     = np.mean(result, axis=0)
    std    = np.std(result + 1e-6, axis=0)
    result = (result - mu) / std
    return result

