'''
Import cfb files and save as npy array files
.. moduleauthor:: Fiona Young
'''

from dxtbx import load
import numpy as np
import glob 

class Cbf2Np():
  '''
  '''

  def __init__(self, file_dir, destination):
    '''
    :param file_dir: str, directory containing files to be converted
    :param destination: str, directory to house new files
    '''
    self.fdir = file_dir
    self.dest = destination

  def cbf_to_npfile(self, cbf_filename, npy_filename=None):
    '''
    convert single cbf file contents to numpy array and save in specified
    directory
    .. automethod:: cbf_to_npfile
    :param cbf_filename: str, input file name
    :param npy_filename: str, output file name
    '''
    if npy_filename is None: # create new file with same name (but npy suffix)
      npy_filename = cbf_filename.replace('.cbf','.npy')

    image = load(self.fdir+cbf_filename)
    data = image.get_raw_data().as_numpy_array()
    np.save(self.dest+npy_filename, data, allow_pickle=False)
  
  def read_cbf_directory(self):
    '''
    read directory and parse each file into a numpy array, save
    .. automethod:: read_cbf_directory
    '''
    # read only cbf files in file_dir
    files = glob.glob(self.fdir+'*.cbf')

    for f in files:
      print(f)
      self.cbf_to_npfile(f.replace(self.fdir,''))
      
  def extract_header(self):
    
