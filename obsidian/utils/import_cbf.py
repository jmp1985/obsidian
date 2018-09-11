'''
Import cfb files and save raw data as npy array files. Additionally, specify a directory containing background measurements to be averaged and saved to the npy data directory. After running, specified destination directory should contain:
  - xxx.npy files in corresponding subdirectories, where xxx is the same as the corresponding cbf file name unless
    otherwise specified
  - background.npy files containing background data averaged across files in
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

def read_header(f, params, file_string=False):
  '''Extract desired parameters from header file. Will not work correctly if params contain any spaces
  
  :param str f: header file path
  :param list params: List of strings, each the name of a parameter found in the header
  :return: dict of param:values where values is a list of all subsequent space separated strings

  Example: read_header(<file>, ['Beam_xy', 'Detector_distance']) will return
  {'Beam_xy' : ['(1251.51,', '1320.12)', 'pixels'], 'Detector_distance':['0.49906','m']}
  '''
  head = f.splitlines() if file_string else open(f, 'r')
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

def progress(n, tot):
  '''Print a progress bar to terminal

  :param int n: number of processed items
  :param int tot: total number of items
  '''
  len = 35
  prog = int(round(len * n/tot))
  if prog == len:
    sys.stdout.write("\r[{0}] {1:.1f}%".format("=" * len, 100*n/tot))    
  elif prog == 0:
    sys.stdout.write("\r[{0}] {1:.1f}%".format("-" * len, 100*n/tot))    
  else:
    sys.stdout.write("\r[{0}{1}{2}] {3:.1f}%".format("=" * (prog-1), ">", "-" * (len-prog), 100*n/tot))    
    sys.stdout.flush()

class Cbf2Np():
  '''Encapusation of cbf to npy conversion process
  '''

  def __init__(self, root, destination, box=False, centre=None, max_res=None):
    '''
    :param str root: directory containing subdirectories of files to be converted
    :param str destination:directory to house new files
    :param bool box: if True, save cropped images up to specified resolution
    :param float max_res: maximum resoltion to crop images to in Angstrom
    :param tuple centre: beam centre in form (beam_y, beam_x) (in pixel coordinates)
    '''
    self.root = root
    self.dest = destination
    self.all_dirs = self.get_dirs() # create dictionary of directory paths and files
    self.box = box
    if box: 
      self.centre = centre # in pixel coords, (row, col)
      self.max_res = max_res

  def get_box(self, header):
    '''Get box indices for cropped image

    :param str header: file path for header txt file
    :returns: start and stop row and col indices
    '''
    rmax = radius_from_res(self.max_res, header)
    if self.centre is None:
      info = read_header(header, ['Beam_xy'])
    centre = tuple(reversed(eval(''.join(info['Beam_xy'][:2])))) if self.centre is None else self.centre   
    y, x = centre[0], centre[1]
    # Return r0, r1, c0, c1
    
    return int(round(y-rmax)), int(round(y+rmax)), int(round(x-rmax)), int(round(x+rmax))
  
  def get_dirs(self):
    '''Scan root for image folders
    '''
    bottom_dirs = {}
    for (dirName, subdirList, fileList) in os.walk(self.root, topdown=True):
      # Collect bottom most directory names and files
      if len(subdirList)==0:
        bottom_dirs[dirName]=fileList
    return bottom_dirs

  def get_npy_filedir(self, cbf_filedir):
    npy_filedir = os.path.join(self.dest, os.path.relpath(cbf_filedir, self.root)) 
    
    # Create destination directory
    if not os.path.isdir(npy_filedir):
      os.makedirs(npy_filedir)
    
    return npy_filedir

  def cbf_to_npfile(self, cbf_filepath, npy_filedir, header=False):
    '''Convert single cbf file contents to numpy array and save in specified 
    directory

    :param str cbf_filename: input file path
    :param str npy_filename: output file name (optional, default same as cbf name)
    :param bool header: if True, will also extract header info from cbf file
    '''
    cbf_filedir, cbf_filename = os.path.split(cbf_filepath)
   
    npy_filename = cbf_filename.replace('.cbf','.npy')
    
    npy_filepath = os.path.join(npy_filedir, npy_filename)
    if header:
      self.extract_header(cbf_filepath, npy_filedir)
 
    # Extract image data:
    image = load(cbf_filepath)
    if self.box:
      h = os.path.join(npy_filedir, "header.txt")
      r0, r1, c0, c1 = self.get_box(h)
      # Extract cropped image data
      data = image.get_raw_data().as_numpy_array()[r0:r1, c0:c1]
    else:  
      data = image.get_raw_data().as_numpy_array()
    np.save(npy_filepath, data, allow_pickle=False)
     
  def read_data_directory(self):
    '''Read directory and parse each file into a numpy array, save
    '''
    for directory in self.all_dirs.keys():
      print("\nWorking through directory {}".format(directory)) 
      header = True
      i = 0
      tot = len(self.all_dirs[directory])
      progress(i, tot)
      npy_filedir = self.get_npy_filedir(directory)
      with open(os.path.join(npy_filedir, 'keys.txt'), 'w') as keys:
        for cbf_file in self.all_dirs[directory]:
          if os.path.splitext(cbf_file)[1] == '.cbf':
            cbf_filepath = os.path.join(directory, cbf_file)
            self.cbf_to_npfile(cbf_filepath, npy_filedir, header=header)
            keys.write(self.get_image_key(cbf_filepath)+'\n')
            header = False # Extract header for first file only
          i+=1
          progress(i, tot)

  def get_image_key(self, cbf_filepath):
    key = 'Image_path'
    head = FormatCBF.get_cbf_header(cbf_filepath)
    stuff = read_header(head, [key], file_string=True)
    ans = stuff['Image_path'][0]
    return '{} {}'.format(cbf_filepath, ans)

  def extract_header(self, cbf_filepath, npy_dir, bg=False):
    '''Assume all headers (mostly) the same for a directory of image files (i.e directory
    contains data from single aquisition experiment)

    :param str cbf_filepath: path to cbf image
    :param str npy_dir: path to destination
    :param bool bg: if True, add 'background' to file name
    '''
    header = open(os.path.join(npy_dir, "{}header.txt".format("bg" if bg else "")), "w")
    header.write(FormatCBF.get_cbf_header(cbf_filepath))
    header.close()
    
  def read_bg_directory(self, bg_dir):
    '''Read background directory and save mean background file to 
    destination directory.

    :param str bg_dir: path to background directory
    '''
    print("Importing background data...")
    # read all files in bg_dir into list of np arrays
    bgData = []
    files = glob.glob(os.path.join(bg_dir,'*.cbf'))
    i = 0
    tot = len(files)
    for f in files:
      img = load(f)
      self.extract_header(f, self.dest, bg=True)
      if self.box:
        h = os.path.join(self.dest, "bgheader.txt")
        r0, r1, c0, c1 = self.get_box(h)
        bgData.append(img.get_raw_data().as_numpy_array()[r0:r1, c0:c1]) # Extract cropped image data
      else: 
        bgData.append(img.get_raw_data().as_numpy_array())
      i += 1
      progress(i, tot)
      
    bgData = np.dstack(bgData)
    bgMean = np.mean(bgData, axis=2)

    np.save(os.path.join(self.dest,"background.npy"), bgMean, allow_pickle=False)

def main(argv):
  data_root = ''
  data_dest = ''
  bg_directory = ''
  kwargs = {}

  help_message = "cbf_to_npy.py \
                  --root <directory containing cbf files (incl subdirs)> \
                  --dest <directory to store npy files in> \
                  -b <background directory> \
                  -c <beam centre in pixels as tuple '(row, col)'> \
                  -r <maximum resoltion to crop images to, in Angstrom>"
  
  try:
    opts, args = getopt.getopt(argv, 'hc:r:b:', ['root=', 'dest='])
  except getopt.GetoptError as e:
    print(e)
    print(help_message)
    sys.exit(2)
  for opt, arg in opts:
    if opt=='-h':
      print(help_message)
      sys.exit()
    elif opt=='--root':
      data_root = arg
    elif opt=='--dest':
      data_dest = arg
    elif opt=='-c':
      kwargs['centre'] = eval(arg) # tuple
    elif opt=='-b':
      bg_directory = arg
    elif opt=='-r':
      kwargs['max_res'] = float(arg)
      kwargs['box'] = True
  assert os.path.exists(data_root), "Invalid data root directory"
  if not os.path.exists(data_dest):
    os.makedirs(data_dest)

  # Instantiate
  do_thing = Cbf2Np(data_root, data_dest, **kwargs)

  # Background data, saved as single averaged file (if specified)
  if bg_directory:
    do_thing.read_bg_directory(bg_directory)

  # Measurement data
  do_thing.read_data_directory()

  print("\nAll done!")

if __name__ == '__main__':
  main(sys.argv[1:])
