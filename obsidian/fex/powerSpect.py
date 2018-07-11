'''
Feature extraction class for analysing image profile power spectra
.. automodule:: powerSpect
.. moduleauthor:: Fiona Young
'''

import numpy as np
import matplotlib.pyplot as plt

class PowerSpect():
  
  '''
  '''

  def __init__(self, trace):
    self.trace = trace
    self.n = trace.size
  
  def spect(self):
    '''
    compute the intensity of spatial frequency components
    '''
    coefs = np.fft.fft(self.trace)/self.n # normalised fft
    freqs = np.fft.fftfreq(self.n)
    
    return np.stack((coefs, freqs))

  def display(self, spect):
    '''
    plot power spectrum
    '''
    fig, ax = plt.subplots()
    ax.plot(spect[1], np.abs(spect[0]))

    return fig, ax

