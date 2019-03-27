"""
.. module:: torchhelpers

This module contains three classes that interact between SCADA data held either
in pandas dataframe or stored in *.csv* files. These data structures return
*torch.tensors* that can be used to load data into neural nets defined using
the pytorch package.

.. moduleauthor:: amartinsson, ZofiaTr
"""
import math
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from .normalisation import min_max, mean_std


class FeedForwardDataset(Dataset):
    """ Class which implements a simple pytorch dataset which interacts with
    the SCADA data and loads it into a PyTorch compatible dataset structure

    :param str/pandas data: the name of a csv file appended with the suffix \
                            or a pandas dataframe
    :param str features: array of feature names
    :param str labels: array of label names
    :param str root_dir: the directory in which the csv file lives
    :param str norm: string defining the method for normalisation
    """

    def __init__(self, data, features, labels, root_dir='../../Data/',
                 norm='min_max', return_index=True):
        """ initialisation of the class """

        # set the names of the features
        self._feature_names = features
        self._label_names = labels
        self.return_index = return_index

        # get a panda dataframe
        if type(data) is str:
            if norm is 'min_max':
                self._landmarks_frame = min_max(filename=data,
                                                root_dir=root_dir)
            elif norm is 'mean_std':
                self._landmarks_frame = mean_std(filename=data,
                                                 root_dir=root_dir)
            else:
                raise ValueError('Cannot find normalisation method! Either it '
                                 'has not been implemented or you need to add '
                                 'it as an option in the FeedForwardDataset '
                                 'class')
        elif type(data) is pd.core.frame.DataFrame:
            self._landmarks_frame = data
        else:
            # never get here
            raise ValueError('data parameter must be either be a string '
                             'or a pandas dataframe object!')

        # make the features torch tensor
        self._features_t = \
            torch.tensor(self._landmarks_frame[self._feature_names].values)\
                 .type(torch.FloatTensor)

        # make the slightly more involved labels tensor
        labels_tmp = \
            torch.tensor(self._landmarks_frame[self._label_names]
                         .values.astype(dtype=np.int64)).type(torch.LongTensor)

        # add other coloumn to labels tensor
        self._labels_t = torch.zeros(size=(len(labels_tmp),
                                           len(labels_tmp[0, :])+1))\
            .type(torch.LongTensor)
        self._labels_t[:, 0:len(labels_tmp[0, :])] = labels_tmp

        # if the first columns are empty add one to the last column
        for i in range(len(self._labels_t)):
            if np.count_nonzero(self._labels_t[i, 0:len(labels_tmp[0, :])])\
                    == 0:
                self._labels_t[i, -1] = 1

        # set the root directory
        self._root_dir = root_dir

        # set the length of the dataset
        self._length = len(self._landmarks_frame)

    def __len__(self):
        """ Returns the length of the dataset when calling len(dataset) """
        return self._length

    def __getitem__(self, idx):
        """ Returns the point at index idx and the points around it as a
        timeframe image.

        :param idx: index of the point
        :return: the element at idx
        """
        # return the elements that were asked for
        if (self.return_index):
            return self._features_t[idx, :], self._labels_t[idx, :], idx

        return self._features_t[idx, :], self._labels_t[idx, :]

    def get_test_train(self, frac=0.33):
        """ Splits the dataset into a test and train set reserving the *frac*
        for the testset. These are returned as samplers which are of
        *torch.utils.data.sampler* type.

        :param frac: fraction to split into the training dataset

        :return: train_sampler, test_sampler
        :rtype: torch.utils.data.sampler, torch.utils.data.sampler
        """
        # get the index and shuffle
        split = np.floor(frac * self._length).astype(np.int64)
        index = np.arange(self._length)

        # shuffle the list of indeices
        np.random.shuffle(index)
        index = list(index)

        # make the set of indecies to sample from
        train_index, test_index = index[split:], index[:split]

        # make index samplers
        sampler = torch.utils.data.sampler.SubsetRandomSampler
        train_sampler = sampler(train_index)
        test_sampler = sampler(test_index)

        # return the samplers
        return train_sampler, test_sampler


class TimeMatrixDataset(FeedForwardDataset):
    """ Class which implements a torch dataset from a file in the root directory.
    This dataset can be used with PyTorch to load data into a neural network.

    :param str/pandas data: the name of a csv file appended with the suffix \
                            or a pandas dataframe
    :param str features: array of feature names
    :param str labels: array of label names
    :param str root_dir: the directory in which the csv file lives
    :param int dim: size along one axis of timepicture, i.e generates dim x dim
    :param str norm: string defining the method for normalisation
    :param bool return_index: geturn also the chosen index by get_item
    """

    def __init__(self, data, features, labels, root_dir='../../Data/',
                 dim=5, norm='min_max', return_index=True):
        """ initialisation of the class """
        # call to the super initialisation
        super().__init__(data=data, features=features, labels=labels,
                         root_dir=root_dir, norm=norm)

        # modify the length
        self._length = len(self._landmarks_frame) - (dim - 1) * (dim + 1)

        # set the dimension
        self._dim = dim
        self.return_index = return_index

    def __getitem__(self, idx):
        """ Returns the point at index idx and the points around it as a
        timeframe image.

        :param idx: index of the point
        :return: the element at idx
        """
        # change index so we don't choose points out of range
        low_index = idx
        cent_index = int(idx + 0.5 * (self._dim - 1) * (self._dim + 1))
        # senare - check that this is correct index!!
        up_index = idx + (self._dim - 1) * (self._dim + 1) + 1

        # get the features and put them into right format
        timeframe = self._features_t[low_index:up_index]\
                        .view(self._dim, self._dim, len(self._feature_names))

        # get the label for this frame
        label = self._labels_t[cent_index, :]
        if self.return_index:
            return torch.transpose(timeframe, 0, 2), label, idx

        if self.return_index:
            return torch.transpose(timeframe, 0, 2), label, idx
        # return everything
        return torch.transpose(timeframe, 0, 2), label


class RegressionDataset(TimeMatrixDataset):
    """ Torch dataset that can be used with regression

    :param str filename: the name of a csv file appended with the suffix
    :param str/pandas data: the name of a csv file appended with the suffix \
                            or a pandas dataframe
    :param str features: array of feature names
    :param str regression: string array of length 1 with regression name
    :param str root_dir: the directory in which the csv file lives
    :param int dim: size along one axis of timepicture, i.e generates dim x dim
    :param str norm: string defining the method for normalisation
    """

    def __init__(self, data, features, regression, root_dir='../../Data/',
                 dim=5, norm='min_max'):
        """ initialisation of the class """
        # check that only one regrassion names was given
        if not len(regression) == 1:
            raise Exception('Too many dimensions was given to regress on '
                            'only one feature can be used!')
        # call to the super constructor
        super().__init__(data=data, features=features,
                         labels=regression, root_dir=root_dir, dim=dim,
                         norm=norm)

        # override how the labels are assigned
        self._labels_t =\
            torch.tensor(self._landmarks_frame[self._label_names].values
                         .astype(dtype=np.float64)).type(torch.FloatTensor)

    def __getitem__(self, idx):
        """ Returns the point at index idx and the points around it as a
        timeframe image.

        :param idx: index of the point
        :return: the element at idx
        """
        # change index so we don't choose points out of range
        low_index = idx
        cent_index = int(idx + 0.5 * (self._dim - 1) * (self._dim + 1))

        # senare - check that this is correct index!!
        up_index = idx + (self._dim - 1) * (self._dim + 1) + 1

        # get the features and put them into right format
        timeframe = self._features_t[low_index:up_index]\
                        .view(self._dim, self._dim, len(self._feature_names))

        # get the label for this frame
        label = self._labels_t[cent_index]

        # return everything
        return torch.transpose(timeframe, 0, 2), label


# senare these should be deleted they are legacy!
def get_torch_matrices(dataframe, features, labels, dim):
    """ returns the relevant matrices of the dataframe using feature and
    label names

    :param pandas dataframe: pandas dataframe of SCADA data
    :param str features: array of feature names
    :param str labels: array of label names
    :param int dim: size along one axis of timepicture, i.e generates \
    dim x dim

    :return: two tensors with features and labels
    :rtype: torch.tensor, torch.tensor
    """
    # how many pictures are we making
    pic_dim = math.floor(dataframe['ts'].count()/(dim * dim))
    print('Making %i pictures and labels...' % pic_dim)

    # make picture holder
    features = np.zeros(shape=(pic_dim, len(features), dim, dim))
    labels = np.zeros(shape=(pic_dim, len(labels)+1))

    # make the dataframe into collection of pictures
    for i in range(pic_dim-1):

        # lower and upper range
        lrange = i * dim * dim
        urange = (i+1) * dim * dim - 1

        # label index
        labelindex = i * dim * dim + math.floor(dim * dim / 2)

        # make pictures
        data = dataframe.loc[lrange:urange, features].values

        # add data to feature vector
        features[i] = np.transpose(data.reshape(dim, dim, len(features)))

        # make the label variable
        label = \
            dataframe.loc[labelindex, labels].values.astype(dtype=np.int64)

        # intepret other column
        if np.count_nonzero(label) == 0:
            label = np.append(label, 1)
        else:
            label = np.append(label, 0)

            labels[i] = label

    return torch.from_numpy(features).type(torch.FloatTensor), \
        torch.from_numpy(labels).type(torch.LongTensor)


def train_test_split(torch_features, torch_labels, frac=0.33):
    ''' splits the two vecotors torch_features and torch_labels into
    fraction *frac* for testing

    :param torch.tensor torch_features: tensor which contains the features
    :param torch.tensor torch_labels: tensor which contains the class labels
    :param float frac: fraction to split into the training dataset

    :return: train_feature, train_labels, test_feature, test_labels
    :rtype: torch.tensor, torch.tensor, torch.tensor, torch.tensor
    '''
    # number of features
    nfeat = math.floor((1-frac) * len(torch_labels))

    # features
    train_feature = torch_features.narrow(0, 0, nfeat)
    test_feature = torch_features[nfeat:len(torch_labels), :, :, :]

    # labels
    train_labels = torch_labels.narrow(0, 0, nfeat)
    test_labels = torch_labels[nfeat:len(torch_labels), :]

    # return
    return train_feature, train_labels, test_feature, test_labels
