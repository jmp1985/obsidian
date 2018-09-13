from setuptools import setup

setup(
    name = "Obsidian",
    version = "0.0.1",
    author = "Fiona Young",
    author_email = "fiona.young@diamond.ac.uk",
    description = "Overcaffeinated Beamline Students Investigate DIffraction And Nanocrystals",
    license = "BSD",
    keywords = "awesome python package",
    packages=[
      'obsidian', 
      'obsidian.learn',
      'obsidian.oimp',
      'obsidian.fex',
      'obsidian.utils',
      'tests',
    ],
    scripts=['bin/obsidian.convnet', 
             'bin/obsidian.import_cbf',
             'bin/obsidian.find_rings',
             'bin/obsidian.process',
             'bin/obsidian.labeller'
    ],
    package_data={'obsidian': ['learn/models/*', 'learn/database.pickle']
    },
    install_requires=[
      'pytest',
      'keras',
      'theano',
      'matplotlib',
      'numpy',
      'sklearn',
      'pandas',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
    ],
)
