
How To Use Obsidian - A Tutorial and Cheat Sheet
************************************************

.. contents::

1. Overview
===========

Obsidian is a package for preprocessing and classifying powder diffraction images based
on the presence of protein rings. Raw data in cbf format first needs to be 
converted to npy format in dials.python before running the rest of obsidian in python3.

Obsidian classification models
------------------------------

Three models are included in obsidian:

======================  =========== =========== =============== === ========  ========
Model name              Sensitivity Specificity Precision (PPV) NPV F1-score  Accuracy
======================  =========== =========== =============== === ========  ========
high_sensitivity_model  96%         82%         81%             96% 88%       88%
high_specificity_model  75%         98%         96%             84% 84%       88%
standard_model          91%         92%         90%             92% 90%       91%
======================  =========== =========== =============== === ========  ========

See `here <https://en.wikipedia.org/wiki/Confusion_matrix>`_ for information on the listed statistics.

If the user wishes to *minimise the number of false positives*, at the cost of a high number
of misses, select high_specificity_model. To *minimise the number of misses*, at the 
cost of a hight number of (possibly time-wasing) false positives, select high_sensitivity_model.
By default obsidian will select the standard model, but the user has the option to specify a different
model or construct their own (detailed below).

2. Data conversion and storage
==============================

Obsidian has a function :mod:`obsidian.import_cbf` which will convert files to npy 
format and crop them down to a relevant resolution if the
following are specified:

* root directory *(required)*: folder containing grid scans to be analysed
* destination directory *(required)*: folder in which to store converted data and 
  background files (will be created if not pre-existing)
* background data directory *(required)*: folder containing blank background scans corresponding to all
  images contained in root.
* beam centre: This must be provided in the order (y, x) or (pixel_row, pixel_column)
* max resolution: maximum resuoltion in Angstroms (reccommended: 7)

.. note::

   1. Normally, obsidian is able to use the beam centre information stored in the image header and does not
      need the information passed explicitely. However, if the user suspects that the actual beam centre 
      deviates significantly from the stored value, they will need to determine it by other means and provide
      the values explicitly as in the example below.
   
   2. It is reccommended to stick to a max resolution value of 7 Angstrom throughout the usage of obsidian,
      in order to maintain consistency with the training data. Specifying max_res at the import_cbf stage will
      significantly reduce file sizes and processing time.

.. warning::

   Make sure destination directory has plenty of disk space (around 30MB per tray)

.. code-block:: console

  $ module load dials
  $ obsidian.import_cbf --root </path/to/data/tray2> --dest <big/disk> -b </path/to/background/data/tray2/g1> -r 7 -c "(1321.16, 1249.94)"

for images cropped to 7 Angstrom, with beam centre at
beam_y = 1321.16, beam_x = 1249.94. Note the quotation marks around beam centre tuple.

You should now have a folder in big/disk corresponding to the root cbf 
directory including all subfolders and a background file.

**All other modules are to be run in python3**

3. Finding protein rings
========================

Once .npy files have been created, preprocessing, feature extraction and 
classification can be performed in a single command with use of the :mod:`obsidian.find_rings` command.

If the images have been pre-cropped (as is reccommended):

.. code-block:: console

   $ obsidian.find_rings -d big/disk/trayX -t 5 --display

* The data directory containing images to be classified must be provided using -d. 
* The top images for each directory will be displayed after classification if the 
  option --display is specified.
* The number of highest ranking images to be displayed can be set with the -t option (default: 10).

Regardless of whether --display is selected, a summary and full results will be saved to the local 
directory in the form of .txt files.

4. Training and updating the obsidian classifier
================================================

4.1 New data
------------

New data must first be converted and processed, then classified using the :mod:`labeller` module.

Convert the data to npy format as described in 2.1.

The directory created by running ``dials.python obsidian.import_cbf`` e.g big/disk/tray2 
should be provided to subsequent modules as image data directoy.

4.1.1 Processing data
+++++++++++++++++++++

.. code-block:: console

   $ obsidian.process -t <top level directory> -d <destination directory>

This will produce xxx_profile.pickle files for each lowest level directoy in big/disk/tray2, corresponding to
each experiment grid scan.

4.1.2 Labelling data
++++++++++++++++++++

Use labeller.py as as in the following example:

.. code-block:: console

   $ obsidian.labeller
   Enter directory containing files to be labelled: /path/to/image/directory
   Enter destination directory for storing classifications: /my/classes
   Classifying directory /path/to/image/directory...
   All file blank (background)? [y/n, default n]: n
   Enter vmax: 600
   Already cropped? [y/n]: y

Vmax refers to the maximum normal pixel value for the images. If entered incorrectly the image contrast
will be wrong making identification difficult. If a bad value is entered, cancel the program and start again.
An easy way to determine vmax is to open an image in adxv and check the colour scale on the control panel:

.. figure:: images/vmax.png
   :align: center

   For this image, select a vmax of 700
   
You will then be presented with each image in turn and asked to classify it. If 'y' entered, the image will
be given the classification 1 (rings visible). Any other key, or simply pressing enter, will classify the
image as 0 (no rings).

In order to proceed to machine learning, you should now have a xxx_classification.pickle file 
corresponding to each xxx_profiles.pickle file.

4.1.3 Combining data and labels
+++++++++++++++++++++++++++++++

Obsidian has a method for creating a database that can be used to update or train a model.

Imagine you have the following files:

::

  some_folder
    └──classes
        ├── xxx_classifications.pickle
        └── yyy_classifications.pickle

  some_other_folder
    └──datadump
        ├── xxx_profiles.pickle
        └── yyy_profiles.pickle

.. note::
   
   These files don't necessarily have to be in separate folders. However, every available 'profiles' file will 
   be included, so make sure any unwanted data are removed from 'datadump'.
   
To build a database and save to a filename of choice (new_database.pickle in the example) do the following:

.. code-block:: python

   >>> from obsidian.utils.data_handling import new_database
   >>> classes_path = 'path/to/some_folder/classes'
   >>> data_path = 'path/to/some_other_folder/datadump'
   >>> database = make_frame(data_path, classes_path, save_path='new_database.pickle')

The above code will build a database out of all available data in 'datadump' and save it to the *current* 
directory as 'new_database.pickle'. To save database to a different directory, specify a full or relative 
path for 'save_path'.

4.2 Preexisting data: Using obsidian.convnet
--------------------------------------------

Once the data has been processed, specifiy its location with the --data option. The argument specified 
after --data could be either a single file or directory containing multiple database files. 
In the latter case, the user will be asked to specify a database from a list of all relevant files found. 
Obsidian only recognises files that fit the pattern '\*database.pickle'.

If no --data argument is specified, obsidian will default to the original Lysozyme dataset.

4.2.1 Updating an existing model
++++++++++++++++++++++++++++++++

.. note::

   While updating an existing model is possible, it's unknown whether this is a good idea. 
   First try testing the model in question on the new data to assess its performance.
   It's unadvisable to update a model on a small amount of data. If necessary, select a small
   number (<10) of epochs for this.

Obsidian comes with 3 pretrained models, trained on the original lysozyme dataset:

* high_sensitivity_model
* high_specificity_model
* standard_model

(see `Obsidian classification models`_ for more details)

To update any of these modules (or a custom model previously trained and saved) with a new set of 
training data, use convnet with the following options:

.. code-block:: console

   $ obsidian.convnet --mode update --name <model name> --data path/to/database  -e <number of epochs>

The result will be an updated model trained on the new data, overwriting the preceding file.

4.2.2 Testing an existing model
+++++++++++++++++++++++++++++++

To test a model on a chosen dataset, use :mod:`obsidian.convnet` in saved mode:

.. code-block:: console

   $ obsidian.convnet --mode saved --name <model name> --data <data to test on>

4.2.3 Training a brand new model
++++++++++++++++++++++++++++++++

To build, train and save new models use the :mod:`learn.convnet` module. New models will be saved in 
obsidian/learn and can be used later on, just be sure to name them carefully! 

.. code-block:: console

   $ obsidian.convnet --name <model name>

The following options are available:

============= ================================= ============================ =================================
Option        Function                          Default                      Example usage
============= ================================= ============================ =================================
--name        name of model                     classifier_model             --name my_new_model
--data        location of training data         *obsidian original database* --data path/to/my_database.pickle
--custom_loss if specified, use weighted        *no custom loss, use*        --custom_loss
              binary crossentropy loss          *binary_crossentropy*
              function
-w            weight to use for weighted        0.5                          -w 0.8
              binary crossentropy loss
              function
-e            number of training epochs         30                           -e 15
-b            batch size                        20                           -b 10
              (note: a smaller batch
              size will increase training
              time.
-n            number of `convolution            6                            -n 3
              layers <https://keras.io/
              layers/convolutional/#conv1d>`_
-p            padding                           same                         -p valid
-o            amount of dropout                 0.3                          -o 0.5
============= ================================= ============================ =================================

*Note on loss weight:*
A value higher than 1 will bias the model towards positive predictions (high false positive rate, low 
false negative rate), a value lower than 1 will bias the model towards negative predictions 
(high false negative rate, low false positive rate).

For more information on the above options see the 
`Keras documentation <https://keras.io/layers/convolutional/#conv1d>`_ or the documentation 
for :mod:`learn.convnet`.

