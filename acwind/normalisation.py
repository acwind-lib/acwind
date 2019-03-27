"""
.. module:: normalisation

This module contain two different function for normalisation of the SCADA data
into the

.. moduleauthor:: amartinsson, ZofiaTr
"""
import pandas as pd
import numpy as np

def min_max(data=None, filename=None, root_dir='../../Data/'):
    """ Performs the min max normalisation:

    .. centered:: :math:`z_i = (x_i - x_{min}) / (x_{max} - x_{min})`

    :param pandas/optional data: a pandas dataframe
    :param str/optional filename: the name of a csv file appended with the\
                                  suffix
    :param root_dir: root directory of the file

    :return: a normalised pandas dataframe
    :rtype: pandas.dataframe
    """
    # make the dataframe
    if filename is not None:
        dataframe = pd.read_csv(root_dir + filename)
    elif data is not None:
        dataframe = data
    elif filename is None and data is None:
        raise ValueError('Must provide either filename or data!')

    # loop over float columns
    for name in dataframe.columns.values:
        if dataframe[name].dtype == np.float64:

            # get min and max
            df_max = dataframe[name].max()
            df_min = dataframe[name].min()

            # write values to new dataframe
            dataframe[name] = (dataframe[name] - df_min)/(df_max - df_min)

    # return the new dataframe
    return dataframe


def mean_std(data=None, filename=None, root_dir='../../Data/'):
    """ Performs the min max normalisation

    .. centered:: :math:`z_i = (x_i - x_{mean}) / x_{std}`

    :param pandas/optional data: a pandas dataframe
    :param str/optional filename: the name of a csv file appended with the\
                                  suffix
    :param str root_dir: root directory of the file

    :return: a normalised pandas dataframe
    :rtype: pandas.dataframe
    """
    # make the dataframe
    if filename is not None:
        dataframe = pd.read_csv(root_dir + filename)
    elif data is not None:
        dataframe = data
    elif filename is None and data is None:
        raise ValueError('Must provide either filename or data!')

    # loop over float columns
    for name in dataframe.columns.values:
        if dataframe[name].dtype == np.float64:

            # calculate mean and standard deviation
            mean = np.mean(dataframe[name].values)
            std = np.std(dataframe[name].values)

            # normalise the data
            dataframe[name] = (dataframe[name] - mean)/std

    # return the new dataframe
    return dataframe
