'''
Create pseudo background file from negative files
'''

import numpy as np
import os, sys
import pickle

def main1(argv):
  
  top = '/media/Elements/obsidian/diffraction_data/180726_small'
  bottoms = {}
  for folder, subdirs, files in os.walk(top):
    if len(subdirs)==0:
      ID = ''.join(folder.split(os.sep)[-3:])
      bottoms[folder] = {'ID':ID, 'files':files}
      print(ID)

  for d in bottoms.keys():

    if os.path.exists('obsidian/datadump/{}_background.npy'.format(bottoms[d]['ID'])):
      print("Background file for {} already exists, skipping".format(bottoms[d]['ID']))
      continue

    try:
      labels = pickle.load(open('/media/Elements/obsidian/diffraction_data/classes/small/{}_classifications.pickle'.format(bottoms[d]['ID']), 'rb'))
      print("making list... numer to test:", len(labels), )
      bg_data = [np.load(sample) for sample in labels.keys() if labels[sample]==0]
    except IOError:
      print("no labels yet")
      continue

    if len(bg_data)!=0:
      print("making bg file")
      try:
        mean_bg_data = np.mean(np.dstack(bg_data), axis=2)
      except MemoryError:
        half = int(round(len(bg_data)/2))
        mean1 = np.mean(np.dstack(bg_data[:half]), axis=2)
        mean2 = np.mean(np.dstack(bg_data[half:]), axis=2)
        mean_bg_data = np.mean(np.dstack((mean1, mean2)), axis=2)
        print("MemoryError handled")
    
      np.save('obsidian/datadump/{}_background.npy'.format(bottoms[d]['ID']), mean_bg_data, allow_pickle=False)


def main2(argv):
  top = '/media/Elements/obsidian/diffraction_data/lysozyme_small'
  bottoms = {}
  for folder, subdirs, files in os.walk(top):
    if len(subdirs)==0 and any(f in folder for f in ('g1', 'f1')):
      ID = ''.join(folder.split(os.sep)[-3:])
      tray = ''.join(folder.split(os.sep)[-4:-2])
      bottoms[folder] = {'ID':ID, 'tray':tray, 'files':files}
  
  for d in bottoms.keys():
    
    if os.path.exists('obsidian/datadump/{}_background.npy'.format(bottoms[d]['ID'])):
      print("Background file for {} already exists, skipping".format(bottoms[d]['ID']))
      continue

    try:
      bg_data = [np.load(os.path.join(d, sample)) for sample in bottoms[d]['files'] if sample.endswith('.npy')]
      print(len(bg_data))
    except IOError as e:
      print("no labels yet",e)
      continue

    if len(bg_data)!=0:
      print("making bg file")
      try:
        mean_bg_data = np.mean(np.dstack(bg_data), axis=2)
      except MemoryError:
        half = int(round(len(bg_data)/2))
        mean1 = np.mean(np.dstack(bg_data[:half]), axis=2)
        mean2 = np.mean(np.dstack(bg_data[half:]), axis=2)
        mean_bg_data = np.mean(np.dstack((mean1, mean2)), axis=2)
        print("MemoryError handled")
    
      np.save('obsidian/datadump/{}_background.npy'.format(bottoms[d]['ID']), mean_bg_data, allow_pickle=False)




if __name__ == '__main__':
  main1(sys.argv[1:])
    

  
