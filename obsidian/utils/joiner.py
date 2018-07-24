'''
join two pickled datadicts together
'''

import pickle

def pickle_get(path):
  pickle_in = open(path, 'rb')
  return pickle.load(pickle_in)

def pickle_put(path, data):
  pickle_out = open(path, 'wb')
  pickle.dump(data, pickle_out)
:x

def main():
  
  ID1 = 'T1a8'
  ID2 = 'T1a8-2'

  path = 'obsidian/datadump/{}_profiles.pickle'
  part1 = pickle_get(path.format(ID1))
  part1.update(pickle_get(path.format(ID2)))
  
  pickle_put(path.format(ID1), part1) 

if __name__ == '__main__':
  main()
