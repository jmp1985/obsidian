'''
Module encapsulating feature extraction to produce 
machine learning input data
'''

from .trace import Trace
import numpy as np
import pickle
import os
import math
from obsidian.utils.data_handling import read_header

def get_params(header):
    '''Return image parameters necessary for calculating a radius in pixels from a given resolution

    :param str header: filepath for header.txt file, containing all image parameters 
    :returns: wavelength, detector distance, pixel size
    '''
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
  '''Calculate radius in pixels of a given maximum resolution. Use to compute rmax for Trace()

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
  '''Feature extraction tool, for extracting and storing mean traces 
  '''
  
  def __init__(self, preparedData):
    '''
    :param preparedData: dataset that has been processed for feature extraction
    '''
    self.data = preparedData
    self.profiles = {}
  
  def mean_traces(self, rmax, nangles, centre=None):
    '''Calculate mean trace vectors for all images in self.data

    :param tuple centre: beam centre in pixel coordinates
    :param int rmax: radius in pixels of data to be extracted, corresponding to the relevant resolution
    :param int nangles: number of lines to average over: the higher the more representative the extracted profile data
    :return: list of mean profile vectors extracted from image collection 
    '''      
    if rmax is None:
      print(("Warning: rmax either not specified or exeeded image dimensions. \
rmax set to half smallest image dimension."))

    angles = np.linspace(89, -89, nangles)
    for name, img in self.data.items():
      tr = Trace(img.shape, angles, centre, rmax)
      self.profiles[name] = tr.mean_trace(img)[1]
    return self.profiles
    
  def dump_save(self, ID, path=None):
    '''Save extracted data to pickle file in obsidian/datadump or specified path

    :param str ID: data batch label for later identification
    :param str path: destination path (if not specified, default is obsidian/datadump)
    '''
    default = 'obsidian/datadump'
    if not os.path.exists(default) and path is None:
      os.makedirs(default)

    path = os.path.join(default if path is None else path, 
                        "{}_profiles.pickle".format(ID))

    profiles_save = open(path, "wb")
    pickle.dump(self.profiles, profiles_save)
    profiles_save.close()
