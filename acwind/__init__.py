"""

This module contains classes and functions for handling SCADA data from wind
turbines. This code was developed between the University of Edinburgh and
DNV GL during a joint project during the summer of 2018. It is meant to compare
several machine learning ideas for automatic classification wind turbine
SCADA data.

It also integrates with the `PyTorch`_ neural net package to utilise both fully
connected and convolutional neural nets. We have written a few modules that
integrate the package

.. _PyTorch: https://pytorch.org/

.. moduleauthor:: amartinsson <anton.martinsson@ed.ac.uk>,\
                  ZofiaTr <zofia.trstanova@ed.ac.uk>
"""

# module name
name = "acwind"

# version number
from .release import __version__

# import all the submodules
from .helpers import *
from .torchhelpers import *

from .alignment import *
from .normalisation import *

from .derivedfeatures import *
from .neuralnetwork import *

from .nnmodel import *
