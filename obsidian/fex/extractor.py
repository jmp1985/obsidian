'''
module encapsulating all feature extraction processes of obsidian to produce the
machine learning input data
'''

from .trace import Trace
import numpy as np
import pickle

class FeatureExtractor():
  
  def __init__(self, preparedData):
    '''
    :param preparedData: dataset that has been processed for feature extraction
    '''
    self.data = preparedData
    self.profiles = {}
  
  def meanTraces(self, centre, rmax, nangles):
  
    angles = np.linspace(89, -89, nangles)
    image_shape = list(self.data.values())[0].shape
    for name, img in self.data.items():
      #import pdb
      #pdb.set_trace()
      tr = Trace(image_shape, angles, centre, rmax)
      self.profiles[name] = tr.meanTrace(img)[1]
    
    return self.profiles
    
  def dump_save(self, ID):
    
    profiles_save = open("obsidian/datadump/{}_profiles.pickle".format(ID), "wb")
    pickle.dump(self.profiles, profiles_save)
    profiles_save.close()
