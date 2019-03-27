"""
.. module:: derivedfeatures

This module contain functions for creating derived features from SCADA data

.. moduleauthor:: amartinsson, ZofiaTr
"""

import numpy as np

def add_relative_weight(dataframe, features, nbins=50):
    """ This function adds a relative weight feature to the pandas dataframe
    i.e it will add a new *rweight* column. It does this by creating a 2D
    histogram in the given features and assigning the height of the histogram
    bin to all the points which fall into that particular bin.

    :param pandas dataframe: a panda dataframe with SCADA data.
    :param str features: an array of two strings with the features names from\
    which the relative weight is derived.
    :param int nbins: number of bins in one direction of the histogram
    """
    # check that only two features were given
    if len(features) is not 2:
        raise ValueError('Must be given the name of two features! But found'
                         ' %1.0d features instead' % len(features))

    # add a new column to the dataframe
    dataframe['rweight'] = 0.0

    # make a histogram from the features
    hist, x, y = np.histogram2d(dataframe[features[0]].values,
                                dataframe[features[1]].values,
                                bins=nbins, normed=True)

    # get all the non-zero elements
    cind, rind = np.nonzero(hist)

    # loop over these bins
    for i in range(len(rind)-1):
        # find the correct elements
        yelements = dataframe.loc[dataframe[features[1]]
                                  .between(y[rind[i]], y[rind[i] + 1])].index
        elements = dataframe.loc[yelements]\
                            .loc[dataframe
                                 .loc[yelements, features[0]]
                                 .between(x[cind[i]], x[cind[i] + 1])].index

        # asign the value
        dataframe.loc[elements, 'rweight'] = hist[cind[i], rind[i]]
