'''
rapidly label images as showing protein rings or not
'''

from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from obsidian.utils.data_handling import read_header
from obsidian.fex.extractor import radius_from_res

def get_rmax():
  h = os.path.join(file_dir, 'header.txt') 
  return int(radius_from_res(7, h))

file_dir = input("\nEnter directory containing files to be labelled: ")

while not os.path.exists(os.path.join(file_dir, 'keys.txt')):
  print("\n Invalid directory or directory does not contain keys.txt.")
  file_dir = input("Try again: ")

dest = input("\nEnter destination directory for storing classifications: ") 

if not os.path.exists(dest):
  os.makedirs(dest)

with open(os.path.join(file_dir, 'keys.txt')) as k:
  keys = {line.split()[0] : line.split()[1] for line in k}

ID = ''.join(file_dir.split(os.sep)[-3:])    
print("ID: ".format(ID))

print("\nClassifying directory {}...... \n".format(file_dir))

rmax = get_rmax()
file_dict = {}
plt.ion()

allblank = (input("All files blanks (background)? [y/n, default n]") == 'y')
vmax = int(input("Enter vmax: "))
cropped = (input("Already cropped? [y/n]:") == 'y')

counter = 1
total = len(keys)
for f in keys:
        
  if not allblank:
    name = os.path.splitext(os.path.basename(f))[0]
    
    img = np.load(f)[1300-rmax:1300+rmax,1200-rmax:1200+rmax] if not cropped else np.load(f)
    print("image {}/{}".format(counter, total))
    plt.imshow(img, cmap='binary', interpolation='None', vmin=-1, vmax=vmax)
    plt.draw()
    decision = input(name+": can you see protein rings? [y/n] ")
      
    clss = int( decision == 'y' ) # 1-> protein; 0-> no protein

  else:
    clss = 0
    
  file_dict[keys[f]] = {'Path':f, 'Class':clss}
  counter += 1

file_dict_save = open(os.path.join(dest, "{}_classifications.pickle".format(ID)), "wb")
pickle.dump(file_dict, file_dict_save)
file_dict_save.close()

print("Classifications saved as '{}' in {}".format("{}_classifications.pickle".format(ID), dest))
