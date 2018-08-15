'''
Create pseudo background file from negative files
'''

import numpy as np
import os, sys
import pickle

def main(argv):
  
  top = '/media/Elements/obsidian/diffraction_data/180726'
  bottoms = {}
  for folder, subdirs, files in os.walk(top):
    if len(subdirs)==0:
      ID = ''.join(folder.split(os.sep)[-3:])
      bottoms[folder] = {'ID':ID, 'files':files}
  
  for d in bottoms.keys():

    try:
      labels = pickle.load(open('/media/Elements/obsidian/diffraction_data/classes/new/{}_classifications.pickle'.format(bottoms[d]['ID']), 'rb'))
      print("making list... numer to test:", len(labels), )
      bg_data = [np.load(sample) for sample in labels.keys() if labels[sample]==0]
    except IOError:
      print("no labels yet")
      continue

    if len(bg_data)!=0:
      try:
        mean_bg_data = np.mean(np.dstack(bg_data), axis=2)
      except MemoryError:
        half = int(round(len(bg_data)/2))
        mean1 = np.mean(np.dstack(bg_data[:half]), axis=2)
        mean2 = np.mean(np.dstack(bg_data[half:]), axis=2)
        mean_bg_data = np.mean(np.dstack((mean1, mean2)), axis=2)
        print("MemoryError handled")
    
      print(mean_bg_data) 
      np.save('obsidian/datadump/{}_background.npy'.format(bottoms[d]['ID']), mean_bg_data, allow_pickle=False)

if __name__ == '__main__':
  main(sys.argv[1:])
    

  
