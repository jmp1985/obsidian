'''
Class encapsulating all image processing, to produce data that can be passed
onto the next stage of Obsidian.
.. automodule:: processor
.. moduleauthor:: Fiona Young
'''

from sbtr_bg import Sbtr_bg
import cv2
from skimage import img_as_ubyte, exposure, restoration
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Processor():
  '''
  '''
  
  def __init__(self, collection, background=None):
    '''
    :param collection: dict of images with filenames to be processed
    '''
    self.collection = collection
    self.processedData = collection
    self.bg = background
    
  def background(self):
    '''
    implement the Sbtr_bg class to remove a background from image collection
    :param bgimage: to be removed
    :returns: modified files
    '''

    sbtrbg = Sbtr_bg(self.bg)
    self.processedData = sbtrbg.subtract(self.processedData)
  
  def rm_artifacts(self, value=600):
    '''
    null pixel values above a reasonable photon count
    :param value: default 600, cutoff value, pixels with higher counts assumed
    artifacts
    '''
    for name, image in self.processedData.items():
      image[image > value] = -1

    self.bg[self.bg > value] = -1

  def find_beam_centre():
    '''
    use circle detection to search for beam position
    '''

  def detect_rings(self):
    '''
    detect rings in image and find their centre coordinates
    '''
    image = img_as_ubyte(self.collection[0])
    output = image.copy()
    
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50,
    param2=30, minRadius=0, maxRadius=0)
    
    if circles is not None:
      circles = np.round(circles[0,:].astype('int'))

      for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (255,0,0), 2)
        cv2.rectangle(output, (x-2,y-2), (x+2, y+2), -1)

    plt.imshow(np.hstack([image, output]),cmap='binary',
    interpolation='nearest', vmin=0, vmax=300)
    plt.show()

  def correct_and_filter(self):
    
    for name, img in self.processedData.items():

      img = exposure.adjust_log(exposure.rescale_intensity(img,
      out_range=(0,1000)), gain=2)
      img = restoration.denoise_tv_chambolle(img, weight=0.2)
      
    fig, ax = plt.subplots()
    ax.imshow(list(self.processedData.values())[5], cmap='binary', interpolation='nearest')
    return fig, ax
    
  def dump_save(self):
    
    print("Pickling...")

    data_save = open("obsidian/datadump/processed.pickle", "wb")
    pickle.dump(self.processedData, data_save, protocol=-1)
    data_save.close()

    print("Pickled!")
