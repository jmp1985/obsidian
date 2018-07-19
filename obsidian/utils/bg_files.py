'''
program for reading in background files for each tray and saving as averaged npy
file
*************************
!! run in dials.python !!
*************************
'''

import numpy as np
import os
from dxtbx import load

mean_backgrounds = []
dest = '/dls/science/users/ywl28659/obsidian/obsidian/datadump'
for n in (4,):
  
  root_dir = '/dls/mx-scratch/adam-vmxm/tray{}'.format(n)
  file_names = []
  for root, dirs, files in os.walk(root_dir, topdown=True):
    dirs[:] = [d for d in dirs if d in ('f1, grid')]
    print("Current dir: {}".format(root))
    for f in files:
      if f.endswith('.cbf'):
        file_names.append(os.path.join(root, f))
  
  if len(file_names) != 0:
    bg_data = []
  
    for f in file_names:
      print(f)
      img = load(f)
      bg_data.append(img.get_raw_data().as_numpy_array())

    try:
      mean_bg_data = np.mean(np.dstack(bg_data), axis=2)
    except MemoryError:
      half = int(round(len(bg_data)/2))
      mean1 = np.mean(np.dstack(bg_data[:half]), axis=2)
      mean2 = np.mean(np.dstack(bg_data[half:]), axis=2)
      mean_bg_data = np.mean(np.dstack((mean1, mean2)), axis=2)
      print("MemoryError handled")

    np.save(dest+'/tray{}_background'.format(n), mean_bg_data, allow_pickle=False)


  
