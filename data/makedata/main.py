'''
Construct and save bogus diffraction data for obsidian
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle_perimeter_aa, circle
from random import randint, uniform, random
from skimage import io as skio
from skimage import img_as_float
import math
from obsidian.utils.imgdisp import ImgDisp

def makeimg(size):
  
  # initialise image array
  img = np.ones(size, dtype=np.float)
  cent = int(size[0]/2)
  
  # parameters
  ncircles = randint(1, 50)
  maxThickness = 80
  
  # draw random circles
  for n in range(ncircles):
    rad = randint(1,500) # radius
    
    if randint(0,1): # draw cicle ...
      rr, cc, val = circle_perimeter_aa(cent, cent, rad, shape=img.shape)
      img[rr, cc] = val
    
    else: # ...or draw disk
      t = uniform(0,maxThickness)
      ri, ci = circle(cent, cent, rad, shape=img.shape)
      ro, co = circle(cent, cent, rad+t, shape=img.shape)
      val = random() # circle darkness
      img[ro, co] -= val
      img[ri, ci] += val
    
    # Images of type float must be between -1 and 1.
    img[img < -1] = -1
    img[img > 1] = 1
  
  return img

def main():
  
  # Parameters
  nimages = 20
  size = (800, 800)
  
  collection = []
  
  # generate images
  for n in range(nimages):
    collection.append(makeimg(size))
  
  # set up image viewer
  imgdisp = ImgDisp(collection)
  imgdisp.disp()
  
  # save images
  for image in range(len(collection)):
    skio.imsave('../fakedata/fakeimg{}.png'.format(image), img_as_float(collection[image]))
  
  # display generated images
  plt.show()

main()
