'''
First Obsidian class: remove background features from a powder diffraction image
.. moduleauthor:: Fiona Young
'''
import numpy as np
from skimage import data, transform
import matplotlib.pyplot as plt

class Sbtr_bg():
  '''
  remove background features from a powder diffraction image
  '''
  
  def __init__(self, bgfile):
    '''
    :param bgfile: data of background image to be subtracted
    '''
    self.bgfile = bgfile
  
  def subtract(self, filelist):
    '''
    Subtract the background file from all data files
    :param filelist: dict of files to be modified
    :returns: newfilelist modified files
    '''
        
    for name, f in filelist.items():
      assert (f.shape == self.bgfile.shape), "Images must be equal size!"
      filelist[name] = (np.subtract(f, self.bgfile))
    
    return filelist
