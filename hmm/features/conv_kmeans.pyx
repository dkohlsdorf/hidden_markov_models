from sklearn.cluster import KMeans
from scipy.fftpack import dct

import numpy as np


class ConvolutionalKMeans:

  def __init__(self, centroids, win_t, win_d):
    self.centroids = centroids
    self.win_t = win_t
    self.win_d = win_d

  @property
  def k(self):
    return len(self.centroids)

  def __getitem__(self, sequence):
    t, d = sequence.shape
    cdef int i,j,k 
    cdef double min_dist = float('inf')
    features = []    
    for i in range(self.win_t, t):
        vector = []
        for k in range(self.k):
            min_dist = float('inf')
            for j in range(self.win_d, d):
                sample   = sequence[i - self.win_t:i, j - self.win_d:j].flatten()
                centroid = self.centroids[k, :]
                distance = np.sum(np.square(sample - centroid))
                if distance < min_dist:
                    min_dist = distance
                idx = j - self.win_d
            vector.append(min_dist)
        features.append(vector)
    features = np.array(features)    

    # normalize
    mu  = np.mean(features, axis=0)
    std = np.std(features,  axis=0)
    for i in range(0, len(features)):
        features[i] = (mu - features[i]) / std
    return features

  @classmethod
  def from_dataset(cls, x, k=10, win_t=10, win_d=30, max_iter=100):
    cdef int n = len(x)
    cdef int i, j, l, t, d
    windows = []    
    for i in range(n):
      sequence = x[i]
      t, d = sequence.shape
      for j in range(win_t, t):
        for l in range(win_d, d):
          windows.append(sequence[j - win_t:j, l - win_d:l].flatten())
    windows = np.stack(windows)
    km = KMeans(k, algorithm='elkan', max_iter=max_iter)
    km.fit(windows)
    return cls(km.cluster_centers_, win_t, win_d)
