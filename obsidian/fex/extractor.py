'''
Module encapsulating all feature extraction processes of obsidian to produce the
machine learning input data
'''

from .trace import Trace
import numpy as np
import pickle
import os
import math
from obsidian.utils.data_handling import read_header

def get_params(header):
    fields = ['Wavelength', 'Detector_distance', 'Pixel_size']
    defaults = False
    try:
      info = read_header(header, fields)
    except Exception as e:
      defaults = True
      raise e
    if defaults:
      wl = 0.96863
      L = 0.49906
      pixel_size = 172*10**(-6)
    else:
      wl = float(info['Wavelength'][0])
      L = float(info['Detector_distance'][0])
      pixel_size = float(info['Pixel_size'][0])
    return wl, L, pixel_size
 
def radius_from_res(max_res, header):
  '''
  Calculate radius in pixels of a given maximum resolution. Use to compute rmax for Trace()

  :param float wl: X-Ray wavelength in Angstrom (e-10 m)
  :param float max_res: Maximum resolution at which user expects to find protein rings, in Angstrom
  :param float distance: Detector distance in meters
  :param float pixel_size: Pixel dimensions in meters
  :return: Radius in pixels
  '''
  wl, L, pixel_size = get_params(header)
  # Convert paramters to m
  l = wl*10**(-10) # Wavelength
  d = max_res*10**(-10) # Resolution
  
  r = L * math.tan(2*math.asin(l/(2*d))) # Radius in meters
  return r/pixel_size # Radius in pixels

class FeatureExtractor():
  
  def __init__(self, preparedData):
    '''
    :param preparedData: dataset that has been processed for feature extraction
    '''
    self.data = preparedData
    self.profiles = {}
  
  def meanTraces(self, centre, rmax, nangles):
    '''
    Calculate mean trace vectors for all images in self.data

    :param tuple centre: beam centre in pixel coordinates
    :param int rmax: radius in pixels of data to be extracted, corresponding to the relevant resolution
    :param int nangles: number of lines to average over: the higher the more representative the extracted profile data
    :return: list of mean profile vectors extracted from image collection 

    .. note::

      meanTraces() is computationally expensive. Select a low nangles value for testing purposes
    '''
    import time
    start = time.time()
    angles = np.linspace(89, -89, nangles)
    image_shape = list(self.data.values())[0].shape
    for name, img in self.data.items():
      tr = Trace(image_shape, angles, centre, rmax)
      self.profiles[name] = tr.meanTrace(img)[1]
    
    secs=time.time()-start
    print("Mean trace extraction: {}:{}min".format(secs//60, secs%60))
    return self.profiles
    
  def dump_save(self, ID, path='default'):
    '''
    Save extracted data to pickle file in obsidian/datadump

    :param str ID: data batch label for later identification
    '''
    
    path = os.path.join('obsidian/datadump' if path == 'default' else path, "{}_profiles.pickle".format(ID))

    profiles_save = open(path, "wb")
    pickle.dump(self.profiles, profiles_save)
    profiles_save.close()
