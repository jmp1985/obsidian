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
      'tests'
    ],
    scripts=[
    ],
    install_requires=[
      'pytest',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
    ],
)
