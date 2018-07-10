'''
Tool for extracting and handling intensity profiles
.. automodule:: trace
.. moduleauthor:: Fiona Young
'''

import math
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
class Trace():
  '''
  Trace extraction and manipulation class
  .. autoclass:: Trace
  '''
  def __init__(self, image):
    '''
    :param image: inputfile as np array
    initialises empty trace array
    '''
    assert(image.shape[0]==image.shape[1]), "image must be square"
    self.img = image
    self.trace = []
    self.w = image.shape[1] # image width is number of columns
    self.h = image.shape[0] # image height is number of rows
    self.cent = self.h/2

  def line(self, angle):
    '''
    line through image
    :param angle: angle w.r.t. horizontal axis in range (-90, 90] deg
    :returns: line coordinates
    '''
    assert(angle > -90 and angle <= 90),  "angle must be in range (-90, 90]"
    
    x0 = self.cent*math.cos(math.radians(angle))
    y0 = self.cent*math.sin(math.radians(angle))
    r0, c0 = self.cent+y0, self.cent-x0 # in pixel coordinates
    r1, c1 = self.cent-y0, self.cent+x0 

    num = 1000
    r, c = np.linspace(r0, r1, num), np.linspace(c0, c1, num)
    # return line coordinates 
    return np.vstack((r, c))

  def readTrace(self, angle):
    ''' 
    extract a single line trace through an image
    :param line: line coordinates as 2d np array
    :returns: line coordinates and image values along line
    '''
    # get line coordinates
    line = self.line(angle)
    # extract image values along line using bilinear interpolation
    vals = map_coordinates(self.img, line, order=1)
    return [line], vals
  
  def meanTrace(self, angles):
    '''
    calculate the mean image profile across specified angles
    :param angles: list of angles to be averaged across
    :returns: traced lines and mean trace values 
    '''
    nAng = len(angles)
    lines = []
    profiles = []
    
    for angle in angles:
      line, vals = self.readTrace(angle)
      lines.append(line)
      profiles.append(vals)
    
    meanVals = np.mean(profiles, axis = 0)
    profiles.append(meanVals)
    
    return lines, profiles

  def display(self, lines, traceVals):
    '''
    display image with line overlaid and resulting trace plot
    '''
    # set up figure gridspace and axes
    fig = plt.figure()
    gs = GridSpec(len(traceVals),2)
    axes = [plt.subplot(gs[:,0])]+[plt.subplot(gs[i,1]) for i in
            range(len(traceVals))]
    
    # show figure in first axis
    axes[0].imshow(self.img, cmap='gray', interpolation='nearest')
    
    # overlay figure with lines indicating profiles
    for line in lines:
      axes[0].plot(line[0][1], line[0][0], 'r')
    
    # plot profile values
    for i in range(len(traceVals)):
      axes[i+1].plot(traceVals[i])
    
    return fig, axes


