'''
general data handling for loading and analysis
'''

import pickle
import os

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


def main():
  
  ID1 = 'T1a8'
  ID2 = 'T1a8-2'

  path = 'obsidian/datadump/{}_profiles.pickle'
  
  join_files(path.format(ID1), path.format(ID2))

if __name__ == '__main__':
  main()
