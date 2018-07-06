'''
Tool for extracting and handling intensity profiles
.. automodule:: trace
.. moduleauthor:: Fiona Young
'''

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
    self.img = image
    self.trace = []
    self.w = image.shape[1] # image width is number of columns
    self.h = image.shape[0] # image height is number of rows
    self.centre = (  

  def getImage(self):
    ''' image getter '''
    return self.img

  def getTrace(self):
    ''' trace getter '''
    return self.trace

  def trace():
    ''' extract a single line trace through an image '''


  


