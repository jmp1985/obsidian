'''
General data handling for loading and analysis
'''
import pandas as pd
import pickle
import os
from glob import glob

def pickle_get(path):
  '''Wrapper function for fetching pickled file contents

  :param str path: path of pickle file
  :returns: contents of pickle file
  '''
  pickle_in = open(path, 'rb')
  return pickle.load(pickle_in)

def pickle_put(path, data):
  '''Wrapper function for dumping data to pickle file

  :param str path: destination path
  :param data: object to be pickled
  '''
  pickle_out = open(path, 'wb')
  pickle.dump(data, pickle_out)

def join_files(end_path, paths):
  '''Combine multiple pickled dictionaries into a single pickle file
  
  :param str end_path: path of target combined file
  :param list paths: list of paths of pickle files to be combined
  '''
  all_data = {}

  for path in paths:
    all_data.update(pickle_get(path))
    os.remove(path)
  pickle_put(end_path, all_data)

def split_data(data_list, chunk_size):
  '''Handle reading and processing data in chunks to avoid the process being killed
  
  :param list data_list: list of items to be split into chunks
  :param int chunk_size: number of items per chunk
  :returns: list of sublists of size chunk_size (except the final sublist, which contains remainder)
  '''
  return [data_list[i:i+chunk_size] for i in range(0, len(data_list), chunk_size)]

def make_frame(datadict):
  '''Create dataframe out of paths contained in datadict

  :param dict datadict: dict in with entries Type:pathlist with Type e.g 'Class', 'Data'
  '''

  data = {}  
  for path in datadict['Data']:
    data.update(pickle_get(path))
  df = pd.DataFrame.from_dict(data, orient='index')
  return df

def read_header(f, params):
  '''Extract desired parameters from header file. Will not work correctly if params contain any spaces
  
  :param str f: header file path
  :param list params: List of strings, each the name of a parameter found in the header
  :return: dict of param:values where values is a list of all subsequent space separated strings

  Example::

    read_header(<header/file/path>, ['Beam_xy', 'Detector_distance'])
  
  returns::
    
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

def update_database(old_database, new_data, save=False):
  '''Add new data to existing database
  
  :param str old_database: path to existing database
  :param list new_data: paths to new data files
  '''
  old = pickle_get(old_database)
  for path in new_data:
    new = pickle_get(path)
    try:
      old.append(new, ignore_index=True)
    except TypeError as e:
      print("Failed to add {}:".format(path), e)
  if save:
    pickle.put(old_database, old)
  return old

def new_database(data_dir, classes_dir, save_path=''):
  
  all_data = {}
  for f in glob(os.path.join(data_dir, '*profiles.pickle')):
    name = os.path.basename(f)
    data = pickle_get(f)
    try:
      labels = pickle_get(os.path.join(classes_dir, name.replace('profiles', 'classifications')))
    except IOError:
      print("No classifications found for {}".format(name))
      continue
    for key in data:
      try:
        data[key].update(labels[key])
      except KeyError:
        continue
    all_data.update(data)
  df = pd.DataFrame.from_dict(all_data, orient='index')
  if save_path:
    pickle_put(save_path, df)
  return df
