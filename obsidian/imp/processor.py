'''
Class encapsulating all image processing, to produce data that can be passed
onto the next stage of Obsidian.
.. automodule:: processor
.. moduleauthor:: Fiona Young
'''

from sbtr_bg import Sbtr_bg

class Processor():
  '''
  .. autoclass:: Processor
  '''
  
  def __init__(self, collection):
  '''
  :param collection: array of images to be processed
  '''
    self.collection = collection

  def background(self, bgimage):
    '''
    implement the Sbtr_bg class to remove a background from image collection
    :param bgimage: to be removed
    :returns: modified files
    '''
    sbtrbg = Sbtr_bg(bgimage)
    processedData = sbtrbg.subtract(self.collection)
    
    return processedData
