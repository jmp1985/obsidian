'''
Feature extraction class for analysing image profile power spectra
.. automodule:: powerSpect
.. moduleauthor:: Fiona Young
'''

import numpy as np
import matplotlib.pyplot as plt

class PowerSpect():
  
  '''
  .. autoclass:: PowerSpect
  '''

  def __init__(self, trace):
    self.trace = trace

  def spect(self):
    '''
    compute the intensity of spatial frequency components
    '''
    n = self.trace.size
    coefs = np.fft.rfft(self.trace)
    freqs = np.fft.fftfreq(n)

    return np.stack((coefs, freqs))

  def display(self, spect):
    '''
    plot power spectrum
    '''
    fig, ax = plt.subplots()
    ax.plot(spect[1], spect[0])

    return fig, ax

