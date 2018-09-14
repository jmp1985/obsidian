'''
Processor module for image processing
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class Processor():
  '''Encapsulation of all image processing, to produce data that can be 
  passed onto the feature extraction stage of Obsidian.
  
  :ivar dict processedData: storage dict for processed images / images to be 
                            processed in form {filepath : image array}
  :ivar array bg: background image data
  '''
  
  def __init__(self, coll, background=None):
    '''
    :param dict coll: dict of filepath : image pairs to be processed
    '''
    self.processedData = coll
    self.bg = background
    
  def background(self, bg=None):
    '''Remove background from image collection
    
    :param bg: Background image to be removed (provide only if not specified on instantiation)
    :returns: modified files
    '''
    assert self.bg is not None or bg is not None, "No background image provided"
    
    bg = self.bg if bg is None else bg

    for name, f in self.processedData.items():
      assert (f.shape == bg.shape), "Background subtraction: images must be equal size!"
      self.processedData[name] = (np.subtract(f, bg))

    if self.bg is None:
      self.bg = bg
  
  def rm_artifacts(self, value=600):
    '''Null pixel values above a reasonable photon count

    :param int value: cutoff value, pixels with higher counts assumed artifacts (default 600)
    '''
    for name, image in self.processedData.items():
      image[image > value] = -1

    if self.bg is not None:
      self.bg[self.bg > value] = -1

  def dump_save(self, ID, path=None):
    '''Save processzed images to pickle file

    :param str ID: identification label
    :param str path: destination path (if None, default is obsidian/datadump)
    '''

    default = 'obsidian/datadump'
    if not os.path.exists(default) and path is None:
      os.makedirs(default)

    path = os.path.join(default if path is None else path, 
                        "{}_processed.pickle".format(ID))
  
    data_save = open(path, "wb")
    pickle.dump(self.processedData, data_save, protocol=-1)
    data_save.close()
