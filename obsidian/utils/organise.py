'''
Create a dataframe that serves as a lookup table for easy access and overview of
all data
'''

import pandas as pd
import os
from glob import glob
import pickle

def make_frame():
  root_path = '/media/Elements/obsidian/diffraction_data'
  file_paths = []
  class_paths = []
  for root, subdirs, files in os.path.walk(root_path):
    file_paths.append(f for f in files if f.endswith('.npy'))
    class_paths.append(f for f in files if f.contains('classifications'))

  for f in file_paths:
    tray_nr = int(f[re.search('tray', f).end()])
    well = int(f[re.search('a\d', f).end()-1])

def rename(dir_name, root_name, which_data):
  
  d = dir_name
  #'obsidian/datadump'
  r = root_name
  #'/media/Elements/obsidian/diffraction_data'

  files = [f for f in glob(os.path.join(d,'*_{}.pickle'.format(which_data)))]

  for f in files:
    data = pickle.load(open(f, 'rb'))
    ID = os.path.basename(f).replace('_{}.pickle'.format(which_data), '')
    tray_nr = ID[1]
    well = ID[2:]
    folder = os.path.join(r, 'tray{}'.format(tray_nr), well, 'grid')

    for key, item in data.items():
      new_key = os.path.join(folder, os.path.basename(key))
      data[new_key] = data.pop(key)

    pickle.dump(data, open(f, 'wb'))

def add_npy_ext(d, pattern):
  for f in glob(os.path.join(d, '*{}'.format(pattern))):
    data = pickle.load(open(f, 'rb'))
    for key, item in data.items():
      if not key.endswith('.npy'):
        new_key=key+'.npy'
        data[new_key] = data.pop(key)
    pickle.dump(data, open(f, 'wb'))


r = '/media/Elements/obsidian/diffraction_data'

which_data = 'classifications'
d = '/media/Elements/obsidian/diffraction_data/classes'

#which_data = 'profiles'
#d = '/dls/science/users/ywl28659/obsidian/obsidian/datadump'

rename(d, r, which_data)


