#!/usr/bin/env python
"""This is the setup script for acwind-lib

To install this library type in the following:

    python setup.py install

Which will install the library in the standard location. If things are not
working contact the authours.
"""

import sys
import os
import shutil
import glob

from distutils.core import setup

long_description = ''' acwind is a python library for automatic
classification for wind energy analytics which has been written entirely in
Python.'''

# this directory
dir_setup = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_setup, 'acwind', 'release.py')) as f:
    # Defines __version__
    exec(f.read())

if __name__ == '__main__':
    setup(name='acwind',
          version=__version__,
          description='automatic classification for wind energy analytics',
          long_description=long_description,
          author='Anton Martinsson, Zofia Trstanova',
          author_email='anton.martinsson [at symbol] ed.ac.uk,\
                        zofia.trstanova [at symbol] ed.ac.uk',
          license='GPL-3.0',
          url='https://github.com/acwind-lib/',
          packages=['acwind'],
          classifiers=[
            # http://pypi.python.org/pypi?%3Aaction=list_classifiers
            'License :: OSI Approved :: GPL License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            ],
          install_requires=[
            'numpy',
            'scipy',
            'scikit-learn',
            'pandas',
            'matplotlib',
            ],
        )
