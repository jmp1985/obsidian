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
    ID = bottoms[d]['ID']
    if os.path.exists('obsidian/datadump/{}_background.npy'.format(ID)):
      print("Background file for {} already exists, skipping".format(ID))
      continue

    try:
      labels = pickle.load(open('/media/Elements/obsidian/diffraction_data/classes/small/{}_classifications.pickle'.format(ID), 'rb'))
      print("making list... numer to test:", len(labels), )
      bg_data = [np.load(sample) for sample in labels.keys() if labels[sample]==0]
    except IOError:
      print("no labels yet for {}".format(ID))
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
    
      np.save('obsidian/datadump/{}_background.npy'.format(ID), mean_bg_data, allow_pickle=False)


def build_bg_files(top, dest):
  bottoms = {}
  
  for folder, subdirs, files in os.walk(top):
    if len(subdirs)==0 and any(f in folder for f in ('g1', 'f1')):
      ID = ''.join(folder.split(os.sep)[-3:])
      tray = ''.join(folder.split(os.sep)[-4:-2])
      bottoms[folder] = {'ID':ID, 'tray':tray, 'files':files}
  
  for d in bottoms.keys():
    
    ID = bottoms[d]['ID']
    tray = bottoms[d]['tray']

    if os.path.exists(os.path.join(dest, '{}_background.npy'.format(ID))):
      print("Background file for {} already exists, skipping".format(ID))
    
    else:
      try:
        bg_data = [np.load(os.path.join(d, sample)) for sample in bottoms[d]['files'] if sample.endswith('.npy')]
        print(len(bg_data))
      except Exception as e:
        print(e)
        continue

      print("making bg file for {}".format(ID))
      try:
        mean_bg_data = np.mean(np.dstack(bg_data), axis=2)
      except MemoryError: # if too much data for a single numpy array, split in half
        half = int(round(len(bg_data)/2))
        mean1 = np.mean(np.dstack(bg_data[:half]), axis=2)
        mean2 = np.mean(np.dstack(bg_data[half:]), axis=2)
        mean_bg_data = np.mean(np.dstack((mean1, mean2)), axis=2)
        print("MemoryError handled")
    
      np.save(os.path.join(dest, '{}_background.npy'.format(ID)), mean_bg_data, allow_pickle=False)
      
    if os.path.exists(os.path.join(dest, '{}_background.npy'.format(tray))):
      print("Background file for {} already exists, skipping".format(tray))

    else:
      try:
        bg_data = [np.load(os.path.join(folder, sample)) for folder in bottoms.keys() 
                                                         for sample in bottoms[folder]['files'] 
                                                         if bottoms[folder]['tray']==tray 
                                                         if sample.endswith('.npy')]
        print(len(bg_data))
      except Exception as e:
        print(e)
        continue

      print("making bg file for {}".format(tray))
      try:
        mean_bg_data = np.mean(np.dstack(bg_data), axis=2)
      except MemoryError: # if too much data for a single numpy array, split in half
        half = int(round(len(bg_data)/2))
        mean1 = np.mean(np.dstack(bg_data[:half]), axis=2)
        mean2 = np.mean(np.dstack(bg_data[half:]), axis=2)
        mean_bg_data = np.mean(np.dstack((mean1, mean2)), axis=2)
        print("MemoryError handled")
    
      np.save(os.path.join(dest, '{}_background.npy'.format(tray)), mean_bg_data, allow_pickle=False)


if __name__ == '__main__':
  top = '/media/Elements/obsidian/diffraction_data/lysozyme_small'
  dest = 'obsidian/datadump'
 
  build_bg_files(top, dest)
    

  
