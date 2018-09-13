'''
Create background file from images classified as blank 
'''

import numpy as np
import os, sys
import pickle

def bg_from_blanks(top, dest):
  '''Make background file by averaging images classified as 
  blanks (without rings)

  :param str top: top level directory containing image folders
  :param str dest: destination for background file
  '''
  
  #top = '/media/Elements/obsidian/diffraction_data/180726_small'
  bottoms = {}
  for folder, subdirs, files in os.walk(top):
    if len(subdirs)==0:
      ID = ''.join(folder.split(os.sep)[-3:])
      bottoms[folder] = {'ID':ID, 'files':files}
      print(ID)

  for d in bottoms.keys():
    ID = bottoms[d]['ID']
    if os.path.exists(os.path.join(dest, '{}_background.npy'.format(ID))):
      print("Background file for {} already exists, skipping".format(ID))
      continue

    try:
      labels = pickle.load(open('/media/Elements/obsidian/classes/small/{}_classifications.pickle'.format(ID), 'rb'))
    except IOError:
      print("no labels yet for {}".format(ID))
      continue
    try:
      bg_data = [np.load(sample) for sample in labels.keys() if labels[sample]==0]
    except Exception as e:
      print(e)
    
    if len(bg_data)!=0:
      print("making bg file")
      try:
        mean_bg_data = np.mean(np.dstack(bg_data), axis=2)
      except MemoryError:
        half = int(round(len(bg_data)/2))
        mean1 = np.mean(np.dstack(bg_data[:half]), axis=2)
        mean2 = np.mean(np.dstack(bg_data[half:]), axis=2)
        mean_bg_data = np.mean(np.dstack((mean1, mean2)), axis=2)
    
      np.save(os.path.join(dest, '{}_background.npy'.format(ID)), mean_bg_data, allow_pickle=False)


def bg_from_scan(top, dest, folders):
  '''Seek out background scan directories and build background files

  :param str top: top level directory containing image folders
  :param str dest: destination for background file
  :param collection folders: list or tuple of folder strings to demarcate background scans (e.g. ('g1', 'f1') )
  '''
  bottoms = {}
  
  for folder, subdirs, files in os.walk(top):
    if len(subdirs)==0 and any(f in folder for f in folders):
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
  folders = ('g1', 'f1')

  bg_from_scan(top, dest, folders)
  bg_from_blanks(top, dest)
