"""
.. module:: neural_network

This module contains neural_network classes.

.. moduleauthor:: amartinsson, ZofiaTr, cmatthews
"""
from __future__ import absolute_import
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
from .helpers import Average

class BasicNN(nn.Module):
    """ Default neural network class.

    """
    def __init__(self):
        super(BasicNN, self).__init__()

    def get_n_params(self):
        """ get the number of all parameters of NN """
        pp = 0
        for p in list(self.parameters()):
            ntmp = 1
            for s in list(p.size()):
                ntmp = ntmp*s
            pp += ntmp
        return pp

    def get_parameters(self):
        """ return all parameters in a list """
        tmp = []
        for p in list(self.parameters()):
            for pk in p.data.numpy().flatten():
                tmp.append(pk)
        return tmp

class CNN(BasicNN):
    """ Class which implements convolutional neural network (CNN):
        two convolutional layers followed by linear layers.

    :param int nFEATURES:  number of features
    :param int nLABELS:  number of labels
    :param int c_size (optional):  convolutional layer size
    :param int linear_size (optional):  number of nodes in linear layers
    :param int kernel_size (optional):  kernel size

    Example::

    >>> import acwind.neural_network as acwnn
    >>> net = acwnn.CNN(len(features), len(labels), linear_size=6, \
        c_size=32, kernel_size=3)
    """

    def __init__(self, nFEATURES, nLABELS, linear_size=6, c_size=32,
                 kernel_size=2):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(nFEATURES, c_size, kernel_size)
        self.conv2 = nn.Conv2d(c_size, c_size, kernel_size)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(c_size, 2*linear_size)
        self.fc2 = nn.Linear(2*linear_size, linear_size)
        self.fc3 = nn.Linear(linear_size, linear_size)
        self.fc4 = nn.Linear(linear_size, nLABELS+1)

    def forward(self, x):
        # run the feature extraction layer
        x = self.feature_extract(x)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def feature_extract(self, x):
        """
        Extracts the outputs from the convolutional layers.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return

class FNN(BasicNN):
    """ Class which implements three hidden layer feed forward neural network

    :param int nFEATURES:  number of features
    :param int nLABELS:  number of labels
    :param int linear_size (optional):  number of noder in linear layers

    Example::

    >>> import acwind.neural_network as acwnn
    >>> net = acwnn.FNN(nFEATURES, nLABELS, linear_size=6)
    """

    def __init__(self, nFEATURES, nLABELS, linear_size=6):
        super(FNN, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(nFEATURES, 2*linear_size)
        self.fc2 = nn.Linear(2*linear_size, linear_size)
        self.fc3 = nn.Linear(linear_size, linear_size)
        self.fc4 = nn.Linear(linear_size, nLABELS+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
