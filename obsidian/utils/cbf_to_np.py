'''
Import cfb files and save raw data as npy array files. Additionally, specify a directory containing background measurements to be averaged and saved to the npy data directory. After running, specified destination directory should contain:
  - xxx.npy files in corresponding subdirectories, where xxx is the same as the corresponding cbf file name unless
    otherwise specified
  - background.npy file, containing background data averaged across files in
    specified background directory
  - header.txt file, containing header information from first read data file
======================================
!! This must be run in dials.python !!
======================================
.. moduleauthor:: Fiona Young
'''

from dxtbx import load
from dxtbx.format.FormatCBF import FormatCBF
import numpy as np
import glob 
import os

class Cbf2Np():
  '''
  '''

  def __init__(self, root, destination):
    '''
    :param file_dir: str, directory containing subdirectories of files to be converted
    :param destination: str, directory to house new files
    '''
    self.root = root
    self.dest = destination
    self.all_dirs = self.get_dirs() # create dictionary of directory paths and files

  def get_dirs(self):
    bottom_dirs = {} # empty directory dict
    for (dirName, subdirList, fileList) in os.walk(self.root, topdown=True):
      #subdirList[:] = [d for d in subdirList if d != 'tray2']
      # collect bottom most directory names and files
      if len(subdirList)==0:
        bottom_dirs[dirName]=fileList
    return bottom_dirs

  def cbf_to_npfile(self, cbf_filepath, npy_filename=None, header=False):
    '''
    convert single cbf file contents to numpy array and save in specified
    directory
    .. automethod:: cbf_to_npfile
    :param cbf_filename: str, input file path
    :param npy_filename: str, output file name (optional)
    '''
    cbf_filedir, cbf_filename = os.path.split(cbf_filepath)
        
    if npy_filename is None: # create new file with same name (but npy suffix)
      npy_filename = cbf_filename.replace('.cbf','.npy')
    
    npy_filedir = os.path.join(self.dest, os.path.relpath(cbf_filedir, self.root))
    #print(npy_filedir, cbf_filedir, npy_filename, cbf_filepath)
    npy_filepath = os.path.join(npy_filedir, npy_filename)
    print("Saving file {}...".format(npy_filepath))

    if not os.path.isdir(npy_filedir):
      os.makedirs(npy_filedir)

    image = load(cbf_filepath)
    data = image.get_raw_data().as_numpy_array()
    np.save(npy_filepath, data, allow_pickle=False)
    
    if header:
      self.extract_header(cbf_filepath, npy_filedir)
  
  def read_data_directory(self):
    '''
    read directory and parse each file into a numpy array, save
    .. automethod:: read_cbf_directory
    '''
    
    for directory in self.all_dirs.keys():
      
      header = True
      for cbf_file in self.all_dirs[directory]:
        
        if os.path.splitext(cbf_file)[1] == '.cbf':

          self.cbf_to_npfile(os.path.join(directory, cbf_file), header=header)
          header = False

  def extract_header(self, cbf_filepath, npy_dir, bg=False):
    '''
    assume all headers the same for a directory of image files (i.e directory
    contains data from single aquisition experiment)
    '''
    header = open(os.path.join(npy_dir, "{}header.txt".format("bg" if bg else "")), "w")
    header.write(FormatCBF.get_cbf_header(cbf_filepath))
    header.close()
    
  def read_bg_directory(self, bg_dir):
    # read all files in bg_dir into list of np arrays
    bgData = []
    files = glob.glob(bg_dir+'.*cbf')
    print(files) 
    for f in files:
      print(f)
      img = load(f)
      bgData.append(img.get_raw_data().as_numpy_array())
      
    bgData = np.dstack(bgData)
    bgMean = np.mean(bgData, axis=2)

    np.save(self.dest+"background.npy", bgMean, allow_pickle=False)

if __name__ == '__main__':

  # root data directory
  data_root = '/dls/mx-scratch/adam-vmxm'

  # destination directory
  data_dest = '/media/Elements/obsidian/diffraction_data'
  assert os.path.exists(data_dest), "Invalid destination path"

  # instantiate
  do_thing = Cbf2Np(data_root, data_dest)

  # measurement data
  do_thing.read_data_directory()

  # background data, saved as single averaged file
  #do_thing.read_bg_directory("data/realdata/tray2/g1/grid/")
