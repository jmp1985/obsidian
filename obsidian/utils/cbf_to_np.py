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
'''

from dxtbx import load
from dxtbx.format.FormatCBF import FormatCBF
import numpy as np
import glob, os, sys, getopt
import math

def read_header(f, params):
  '''
  Extract desired parameters from header file. Will not work correctly if params contain any spaces
  
  :param str f: header file path
  :param list params: List of strings, each the name of a parameter found in the header
  :return: dict of param:values where values is a list of all subsequent space separated strings

  Example: read_header(<file>, ['Beam_xy', 'Detector_distance']) will return
  {'Beam_xy' : ['(1251.51,', '1320.12)', 'pixels'], 'Detector_distance':['0.49906','m']}
  '''
  head = open(f, 'r')
  info = {}
  # Read header file line by line
  for l in head:
    if any(param in l for param in params):
      p = [param for param in params if param in l][0]
      info[p] = l.split(' ')[2:] # extract all info following parameter keyword
  return info

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

class Cbf2Np():
  '''
  '''

  def __init__(self, root, destination, box=False, centre=None, max_res=None):
    '''
    :param file_dir: str, directory containing subdirectories of files to be converted
    :param destination: str, directory to house new files
    '''
    self.root = root
    self.dest = destination
    self.all_dirs = self.get_dirs() # create dictionary of directory paths and files
    self.box = box
    if box: 
      self.centre = centre # in pixel coords, (row, col)
      self.max_res = max_res

  def get_box(self, header):
    '''
    Get box indices for cropped image

    :param str header: file path for header txt file
    :returns: start and stop row and col indices
    '''
    rmax = radius_from_res(self.max_res, header)
    y, x = self.centre[0], self.centre[1]
    # Return r0, r1, c0, c1
    
    return int(round(y-rmax)), int(round(y+rmax)), int(round(x-rmax)), int(round(x+rmax))
  
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
    Convert single cbf file contents to numpy array and save in specified
    directory

    :param cbf_filename: str, input file path
    :param npy_filename: str, output file name (optional)
    '''
    cbf_filedir, cbf_filename = os.path.split(cbf_filepath)
   
    if npy_filename is None: # create new file with same name (but npy suffix)
      npy_filename = cbf_filename.replace('.cbf','.npy')
    
    npy_filedir = os.path.join(self.dest, os.path.relpath(cbf_filedir, self.root))
    npy_filepath = os.path.join(npy_filedir, npy_filename)

    if not os.path.isdir(npy_filedir):
      os.makedirs(npy_filedir)

    if header:
      self.extract_header(cbf_filepath, npy_filedir)
 
    # Extract image data:
    image = load(cbf_filepath)
    if self.box:
      h = os.path.join(npy_filedir, "header.txt")
      r0, r1, c0, c1 = self.get_box(h)
      data = image.get_raw_data().as_numpy_array()[r0:r1, c0:c1] # Extract cropped image data
    else:  
      data = image.get_raw_data().as_numpy_array()
    np.save(npy_filepath, data, allow_pickle=False)
     
  def read_data_directory(self):
    '''
    Read directory and parse each file into a numpy array, save
    '''
    
    for directory in self.all_dirs.keys():
      print("Working through directory {}".format(directory)) 
      header = True

      for cbf_file in self.all_dirs[directory]:
        if os.path.splitext(cbf_file)[1] == '.cbf':
          self.cbf_to_npfile(os.path.join(directory, cbf_file), header=header)
          header = False

  def extract_header(self, cbf_filepath, npy_dir, bg=False):
    '''
    Assume all headers the same for a directory of image files (i.e directory
    contains data from single aquisition experiment)
    '''
    header = open(os.path.join(npy_dir, "{}header.txt".format("bg" if bg else "")), "w")
    header.write(FormatCBF.get_cbf_header(cbf_filepath))
    header.close()
    
  def read_bg_directory(self, bg_dir):
    '''
    '''
    # read all files in bg_dir into list of np arrays
    bgData = []
    files = glob.glob(bg_dir+'.*cbf')
    print(files) 
    for f in files:
      img = load(f)
      bgData.append(img.get_raw_data().as_numpy_array())
      
    bgData = np.dstack(bgData)
    bgMean = np.mean(bgData, axis=2)

    np.save(os.path.join(self.dest,"background.npy"), bgMean, allow_pickle=False)

def main(argv):
  
  data_root = ''
  data_dest = ''
  kwargs = {}
  try:
    opts, args = getopt.getopt(argv, 'hc:r:', ['root=', 'dest='])
  except getopt.GetoptError as e:
    print(e)
    print("cbf_to_npy.py --root <directory containing cbf files (incl subdirs)> --dest <directory to store npy files in>")
    sys.exit(2)
  for opt, arg in opts:
    if opt=='--root':
      data_root = arg
    elif opt=='--dest':
      data_dest = arg
    elif opt=='-c':
      kwargs['centre'] = eval(arg) # tuple
    elif opt=='-r':
      kwargs['max_res'] = float(arg)
      kwargs['box'] = True
  
  assert os.path.exists(data_root), "Invalid data root directory"
  assert os.path.exists(data_dest), "Invalid destination path"

  # instantiate
  do_thing = Cbf2Np(data_root, data_dest, **kwargs)

  # measurement data
  do_thing.read_data_directory()

  # background data, saved as single averaged file
  #do_thing.read_bg_directory("data/realdata/tray2/g1/grid/")

if __name__ == '__main__':
  main(sys.argv[1:])

  
