Setting up
**********

Installation
============

The obsidian package can be downloaded from `<https://github.com/jmp1985/obsidian>`_.

To use obsidian for testing and developing, run setup.py from the installation directory;

.. code-block:: console

   $ python setup.py develop --user

Make sure that path ~/.local/bin is defined in ~./bashrc_local or similar, to enable the execution
of python commands.

Usage
=====

Obsidian uses Keras through the theano backend. Since Keras uses Tensorflow backend by default, the user
must first change the backend used by keras. To do this, open the keras.json file (probably in ~/.keras or a
similar location) and change the backend argument from 'tensorflow' to 'theano'.

Obsidian is a python3 package (with the exception of obsidian.import_cbf). To activate python3 on a Diamond 
Light Source machine, execute the following:

.. code-block:: console

   $ module load python/3
   $ source activate python3

To use obsidian.import_cbf, first load dials.

.. code-block:: console

   $ module load dials
   $ obsidian.import_cbf ...
