'''
Tool for extracting and handling intensity profiles
'''

import math
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
class Trace():
  '''
  Trace extraction and manipulation class

  :ivar int w: image width in pixels
  :ivar int h: image height in pixels
  :ivar list lines: array of line coordinates along which image values will be extracted
  '''
  def __init__(self, image_shape, angles, centre=None, rmax=None):
    '''
    Initialise array of line coordinates corresponding to `angles`. Set instance variables width, height, rmax and lines

    :param tuple image_shape: image dimensions in pixels    
    :param tuple centre: (row, col) pixel coordinates of beam centre. If not provided image assumed to be square and centred
    :param int rmax: radius in pixels of data to be extracted, corresponding to maximum resolution of interest
    '''
    self.w = image_shape[1] # image width is number of columns
    self.h = image_shape[0] # image height is number of rows
    
    if centre is None:
      assert(image_shape[0]==image_shape[1]), "image must be square if centre coordinates not specified"
      self.cent = (self.h/2, self.w/2)
    else:
      self.cent = centre
      
    if rmax is None:
      self.rmax = min(*self.cent, self.w-self.cent[1], self.h-self.cent[0])
    else:
      self.rmax = rmax
    
    self.lines = [ self.line(angle) for angle in angles ]

  def line(self, angle):
    '''
    Calculate coordinates of a single line through image

    :param angle: angle w.r.t. horizontal axis in range (-90, 90] deg
    :returns: line coordinates
    '''
    assert(angle > -90 and angle <= 90),  "angle must be in range (-90, 90]"
    
    x0 = self.rmax*math.cos(math.radians(angle))
    y0 = self.rmax*math.sin(math.radians(angle))
    r0, c0 = self.cent[0]+y0, self.cent[1]-x0 # in pixel coordinates
    r1, c1 = self.cent[0]-y0, self.cent[1]+x0 

    num = self.w 
    r, c = np.linspace(r0, r1, num), np.linspace(c0, c1, num)
    # return line coordinates 
    return np.vstack((r, c))

  def readTrace(self, line, img):
    ''' 
    Extract image values along a single line

    :param np.array line: line coordinates as 2d np array
    :param img: input image
    :returns: line coordinates and image values along line
    '''
    # extract image values along line using bilinear interpolation
    vals = map_coordinates(img, line, order=3)
    return vals
  
  def meanTrace(self, img):
    '''
    Calculate the mean image profile across all angles
    
    :param img: input image
    :returns: individual traces and mean trace values 
    '''
    traces = []
    
    for line in self.lines:
      traces.append(self.readTrace(line, img))
    
    meanVals = np.mean(traces, axis = 0)
    
    return traces, meanVals

  def display(self, img, traces, meanVals):
    '''
    Display image with lines overlaid and resulting trace plots
    
    :param img: input image
    :param list traces: individual traces from each line
    :param list meanVals: values averaged across all lines 
    :return: pyplot figure and axes objects
    '''
    # set up figure gridspace and axes
    fig = plt.figure()
    gs = GridSpec(len(traces),2)
    axes = [plt.subplot(gs[0:-4,0]),
    plt.subplot(gs[-4:,0])]+[plt.subplot(gs[i,1]) for i in range(len(traces))]
    
    # show figure in first axis
    axes[0].imshow(img, cmap='binary', interpolation='nearest', vmin=-1, vmax=300)
    
    # overlay figure with lines indicating profiles
    for line in self.lines:
      print(line)
      axes[0].plot(line[0][1], line[0][0], 'r')
    
    # plot mean trace below figure
    axes[1].plot(meanVals)

    # plot trace values
    for i in range(len(traces)):
      axes[i+2].plot(traces[i])
    
    return fig, axes
