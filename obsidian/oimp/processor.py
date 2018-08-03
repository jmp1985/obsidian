'''
'''

from obsidian.oimp.sbtr_bg import Sbtr_bg
import cv2
from skimage import img_as_ubyte, exposure, restoration
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Processor():
  '''
  Class encapsulating all image processing, to produce data that can be passed onto the feature extraction stage of Obsidian.
  '''
  
  def __init__(self, coll, background=None):
    '''
    :param collection: dict of filename : image pairs to be processed
    '''
    self.collection = coll
    self.processedData = coll
    self.bg = background
    
  def background(self, bg=None):
    '''
    Implement the :class:`Sbtr_bg` class to remove a background from image collection
    
    :param background: to be removed (provide only if not specified on instantiation)
    :returns: modified files
    '''
    assert self.bg is not None or bg is not None, "No background image provided"
    
    sbtrbg = Sbtr_bg(self.bg) if bg is None else Sbtr_bg(bg)
    self.processedData = sbtrbg.subtract(self.processedData)
  
  def rm_artifacts(self, value=600):
    '''
    Null pixel values above a reasonable photon count

    :param int value: cutoff value, pixels with higher counts assumed artifacts (default 600)
    '''
    for name, image in self.processedData.items():
      image[image > value] = -1

    if self.bg is not None:
      self.bg[self.bg > value] = -1

  def correct_and_filter(self):
    '''
    Adjust contrast and noise levels in images
    '''
    for name, img in self.processedData.items():

      img = exposure.adjust_log(exposure.rescale_intensity(img,
      out_range=(0,1000)), gain=2)
      img = restoration.denoise_tv_chambolle(img, weight=0.2)
      
    fig, ax = plt.subplots()
    ax.imshow(list(self.processedData.values())[5], cmap='binary', interpolation='nearest')
    return fig, ax
    
  def dump_save(self, ID):
    '''
    Save processed images to pickle file

    :param str ID: identification label
    '''
    
    print("Pickling...")

    data_save = open("obsidian/datadump/{}_processed.pickle".format(ID), "wb")
    pickle.dump(self.processedData, data_save, protocol=-1)
    data_save.close()

    print("Pickled!")
