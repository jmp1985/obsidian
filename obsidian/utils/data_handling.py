'''
General data handling for loading and analysis
'''
import pandas as pd
import pickle
import os
from glob import glob

def pickle_get(path):
  pickle_in = open(path, 'rb')
  return pickle.load(pickle_in)

def pickle_put(path, data):
  pickle_out = open(path, 'wb')
  pickle.dump(data, pickle_out)

def join_files(end_path, paths):
  all_data = {}

  for path in paths:
    all_data.update(pickle_get(path))
    os.remove(path)
  pickle_put(end_path, all_data)

def split_data(data_list, chunk_size):
  '''
  handle reading and processing data in chunks to avoid the process being killed
  '''
  return [data_list[i:i+chunk_size] for i in range(0, len(data_list), chunk_size)]

def make_frame(datadict):
  '''
  :param dict datadict: dict in with entries Type:pathlist with Type e.g 'Class', 'Data'
  '''
  data = {}
  for path in datadict['Data']:
    data.update(pickle_get(path))
  classes = {}
  for path in datadict['Class']:
    classes.update(pickle_get(path))

  df = pd.DataFrame([[key, data[key], classes[key]] for key in data.keys()], columns =['Path', 'Data', 'Class'])
  print(df)
  '''
  for name, l in datadict.items():
    all_files = {}
    for path in l:
      all_files.update(pickle_get(l))
    df = pd.DataFrame.from_dict(all_files, orient='index') 
  '''
  return df
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


def main1():
  
  ID1 = 'T1a8'
  ID2 = 'T1a8-2'

  path = 'obsidian/datadump/{}_profiles.pickle'
  
  join_files(path.format(ID1), path.format(ID2))

def main2():
  r = '/media/Elements/obsidian/diffraction_data'

  which_data = 'classifications'
  d = '/media/Elements/obsidian/diffraction_data/classes'

  #which_data = 'profiles'
  #d = '/dls/science/users/ywl28659/obsidian/obsidian/datadump'

  rename(d, r, which_data)

if __name__ == '__main__':
  main1()
  #main2()
