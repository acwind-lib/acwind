"""
.. module:: helpers

This module contains a few different functions that are designed to help
with the development of new code. These are general helper functions to
do with visualisation and interaction with both torch tensors and pandas
dataframes.

.. moduleauthor:: amartinsson, ZofiaTr
"""
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def create_confusion_matrix_plot(cmatrix, labels, size=5, cmap=plt.cm.Blues):
    """ This function creates and prints the values of the confusion matrix.

    :param float cmatrix: the confusion matrix as array of floats
    :param int classes: list of names of the different classes
    :param cmap cmap: the colourmap of the confusion matrix

    :rtype: matplotlib.pyplot.figure
    """
    # rename the labels and add other class
    classes_confusion_matrix = labels + ['other']

    # normalise the matrix
    cmatrix_cache = cmatrix.astype('float')
    #cmatrix = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]
    cmatrix_sum = cmatrix.sum(axis=1)[:, np.newaxis]
    zero_sum = cmatrix_sum == 0
    cmatrix_sum[zero_sum] = 1
    cmatrix = cmatrix.astype('float') / cmatrix_sum

    # plot the confusion matrix
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(size, size))
    axs.imshow(cmatrix, interpolation='nearest', cmap=cmap)
    #plt.figure(figsize=(size, size))

    # set title and colourbar
    axs.set_title('Confusion Matrix', fontsize=4*size)

    # set the ticks and rotate them
    tick_marks = np.arange(len(classes_confusion_matrix))
    plt.xticks(tick_marks, classes_confusion_matrix, rotation=45,
               fontsize=2.5*size)
    plt.yticks(tick_marks, classes_confusion_matrix, fontsize=2.5*size)
    #axs.set_xticks(tick_marks, classes_confusion_matrix)
    #axs.set_yticks(tick_marks, classes_confusion_matrix)

    # format the text
    fmt = '.1f'
    fmt_int = '.0f'

    thresh = np.nanmax(cmatrix) / 2.
    #thresh = cmatrix.max() / 2.

    # add text to plot
    for i, j in zip(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        if zero_sum[i] == False:
            axs.text(j, i, format(cmatrix[i, j]*100, fmt)
                     + '%' +
                     '\n (' + format(cmatrix_cache[i, j], fmt_int) + ')',
                     horizontalalignment="center", verticalalignment='center',
                     color="white"
                     if cmatrix[i, j] > thresh
                     else "black", fontsize=2.0*size)
        else:
            axs.text(j, i, '-',
                     horizontalalignment="center", verticalalignment='center',
                     color="black", fontsize=2.0*size)

    # adjust labels and layout
    fig.tight_layout()
    axs.set_ylabel('True label', fontsize=3.5*size)
    axs.set_xlabel('Predicted label', fontsize=3.5*size)


def gaussian(x, A, mu, sigma):
    """ Function which returns a gaussian of the parameters given.

    :param float x: argument of the gaussian
    :param float A: the amplitude
    :param float mu: the mean of the gaussian
    :param float sigma: the standard deviation of the gaussian

    :return: :math:`A \exp[-(x-\mu)^2 / 2\sigma^2 ]`
    :rtype: np.float
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def get_frac(dataframe, features, labels=None, frac=0.1):
    """ Function which returns the fraction of feature names and label names
    as pandas dataframes

    :param pandas dataframe: pandas dataframe
    :param str features: array of feature names
    :param str labels: array of label names
    :param float frac: fraction of dataframe to use

    :return: feauters, optional if not None labels
    :rtype: pandas.dataframe, pandas.dataframe
    """
    # sample the dataframe
    datafrac = dataframe.sample(frac=frac)

    # get the labels and features
    _features = datafrac[features]

    # return what was asked for
    return _features if labels is None else _features, datafrac[labels]


def label_color(labels):
    """ function which takes *pandas.dataframe* of labels
    and returns an int array of label colors

    :param labels: 2D array of boolean values

    :return: clabel a colour label
    :rtype: int
    """
    # get the label colours
    clabel = np.zeros(len(labels))
    clabel[np.argwhere(labels.values == True)[:, 0]] \
        = np.argwhere(labels.values == True)[:, 1] + 1

    # return the colour label
    return clabel


def plot_confusion_matrix(true_label, pred_label, labels, size=5):
    """ plots the confusion matrix in a plot and returns it

    :param int true_label: array of indexes corresponding to true class
    :param int pred_label: array of indexes corresponding to predicted class
    :param str labels: array of label names
    :param int size: scales the figuresize

    :rtype: matplotlib.pyplot.figure
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(
        true_label, pred_label, labels=range(len(labels)+1))
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    fig = create_confusion_matrix_plot(cnf_matrix, labels, size=size)

    # return the figure
    return fig


def plot_pca_decomp(extracted, true, predictions, size=5):
    """ Function which plots the pca decomposition of the
    extracted features.

    :param float extracted: multidimensional array which is the output of the \
    feature extraction layer in a Neural Net
    :param int true: array of indexes corresponding to true class
    :param int predictions: array of indexes corresponding to predicted class
    :param int size: scales the figuresize

    :rtype: matplotlib.pyplot.figure
    """
    # reshape into 2D array
    extracted_reshape = extracted.reshape(len(extracted),
                                          np.prod(np.shape(extracted)[1:]))

    # make the pca model and extract the transform
    pca = PCA(n_components=2)
    x_new = pca.fit_transform(extracted_reshape)

    # make the plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2*size, size))

    # get the min and max for the colors
    c_max = true.max()
    c_min = true.min()

    # plot the actual values
    axs[0].scatter(x_new[:, 0], x_new[:, 1], c=true, vmin=c_min, vmax=c_max,
                   edgecolors='k', s=3.0, linewidths=0.1)
    axs[0].set_title('true')

    # plot the predicted values
    axs[1].scatter(x_new[:, 0], x_new[:, 1], c=predictions, vmin=c_min,
                   vmax=c_max, edgecolors='k', s=3.0, linewidths=0.1)
    axs[1].set_title('predicted')

    # return the plot
    return fig


def feature_plot(data=None, dataframe=None, features=None, frac=0.1,
                 labels=None, predictions=None, size=5):
    """ Plots a 2D plot of the SCADA features

    :param ndarray/optional data: an 2D array of data
    :param pandas/otional dataframe: a dataframe with all data
    :param str/optional features: string of the feature names
    :param float/optional frac: proportion of pandas array to plot
    :param str/ndarray labels: string of the labels names or ndarray of class
                               indicies.
    :param ndarray/optional predictions: ndarray of indicies of predictions
    :param int size: scaling of the plots

    :rtype: matplotlib.pyplot.figure
    """

    if data is not None:
        # set the signals
        signals = data[0: np.floor(frac * len(labels)), :]
        classes = labels[0: np.floor(frac * len(labels))]

        # get the min and max for the colors
        c_max = labels.max()
        c_min = labels.min()

    if dataframe is not None:
        # get the fraction which we require
        signals, classes = get_frac(dataframe=dataframe, features=features,
                                    frac=frac, labels=labels)

        # convert the signals and classes
        signals = signals.values
        classes = label_color(classes)

    elif dataframe is None and data is None:
        raise ValueError('Must provide either dataframe or data!')

    # check if predictions were given
    if predictions is None:
        # make the plot
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(size, size))

        axs.scatter(signals[:, 0], signals[:, 1],
                    c=classes, edgecolors='k')

        axs.set_title('Feature Plot')
        axs.set_xlabel(features[0])
        axs.set_ylabel(features[1])

    else:
        # shorten the predictions
        predictions_frac = predictions[0: np.floor(frac * len(predictions))]

        # make the plot
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2*size, size))

        axs[0].scatter(signals[:, 0], signals[:, 1], vmin=c_min, vmax=c_max,
                       c=classes, edgecolors='k')

        axs[1].scatter(signals[:, 0], signals[:, 1], vmin=c_min, vmax=c_max,
                       c=predictions_frac, edgecolors='k')

        axs[0].set_title('True')
        axs[0].set_xlabel(features[0])
        axs[0].set_ylabel(features[1])

        axs[1].set_title('Predictions')
        axs[1].set_xlabel(features[0])
        axs[1].set_ylabel(features[1])

    # return the figure
    return fig


def getloss(data, features, labels=None, truth=None, preds=None, nbins=100, exclude=['normal'], print_summary=True):
    """ Estimates the power loss from given data, and compares truth to prediction

    :param ndarray data: a 2D array of data
    :param str/ndarray features: the list of features in the data
    :param ndarray/optional labels: the list of labels for the classes
    :param ndarray/optional truth: the true class labels for the data
    :param ndarray/optional preds: the predicted class labels for the data
    :param int/optional nbins: the number of bins to compute the baseline over
    :param ndarray/optional exclude: the list of normal classes to exclude from the loss
    :param bool/optional print_summary: print summary statistics about the data

    :return dy the loss for each data point
    :rtype: ndarray
    """
    df = pd.DataFrame(data=data, columns=features)

    xx = df['nws'].values
    yy = df['pwr'].values
    aa, bb, cc = np.histogram2d(xx, yy, bins=nbins)
    cc = np.linspace(cc[0], cc[-1], nbins)

    xii = 1
    bx = np.zeros(nbins)
    by = np.zeros(nbins)
    for ii in range(nbins):
        bx[ii] = (bb[ii+1] + bb[ii])/2
        xii = xii+np.argmax(aa[ii, (xii):])
        by[ii] = cc[xii]  # + cc[xii+1])/2

    yy_baseline = np.interp(xx, bx, by)
    dy = (yy_baseline - yy) / np.sum(yy_baseline)
    if (labels is None):
        return dy, [bx, by]

    labels_all = labels + ['other']

    nlabels = len(labels_all)

    fmtstr = '{0}:  True loss = {1},  Predicted loss = {2},  Error = {3}  ({4}%)'

    true_tot_loss = 0
    pred_tot_loss = 0

    for ii in range(nlabels):

        cname = labels_all[ii]

        predloss = np.sum(dy[preds == ii])
        trueloss = np.sum(dy[truth == ii])

        if (not (cname in exclude)):
            true_tot_loss += trueloss
            pred_tot_loss += predloss

        if (np.abs(trueloss) > 0):
            dloss = np.abs(100*(trueloss-predloss)/trueloss)
        else:
            dloss = 0

        tloss_str = '{0:.3e}'.format(trueloss).rjust(10)
        ploss_str = '{0:.3e}'.format(predloss).rjust(10)
        eloss_str = '{0:.3e}'.format(np.abs(trueloss-predloss)).rjust(10)
        dloss_str = '{0:.3f}'.format(dloss).rjust(7)
        if (print_summary):
            print(fmtstr.format(cname.ljust(12), tloss_str,
                                ploss_str, eloss_str, dloss_str))

    aa = np.sum(true_tot_loss)
    bb = np.sum(pred_tot_loss)
    err = (aa-bb)
    err_pc = 100*np.abs((aa-bb)/aa)
    if (print_summary):
        print('')
        print('Total     loss with exclusions = ' + str(aa))
        print('Predicted loss with exclusions = ' + str(bb))
        print('Error = {0:.3e}   ({1:.3f}%)'.format(err, err_pc))

    return dy, [bx, by], [aa, bb]


class Average():
    """
    Compute the average of an observable.

    :param initial_value: initial value that will define the type of the sample.
    :param blockSize:  number of blocks in the block average, default is 10000
    """

    def __init__(self, initial_value, blockSize=10000):

        self.initial_value = initial_value
        self.current_block_index = 0
        self.blockSize = blockSize
        self.current_index_in_current_block = 0

        self.current_block_sum = initial_value
        self.sum_of_block_averages = initial_value

    def addSample(self, item):
        """
        Add sample to the average.

        :param item: new sample of the same type as initial_value
        """

        self.current_block_sum = self.current_block_sum + item
        self.current_index_in_current_block = self.current_index_in_current_block + 1

        if self.current_index_in_current_block == self.blockSize:
            # add the block average to the sum of block averages
            self.current_block_average = self.current_block_sum / \
                float(self.blockSize)
            self.sum_of_block_averages = self.sum_of_block_averages + \
                self.current_block_sum / float(self.blockSize)

           # begin a new block
            self.current_block_sum = 0.0 * self.initial_value
            self.current_index_in_current_block = 0
            self.current_block_index = self.current_block_index + 1

    def getAverage(self):
        """
        Returns the accumulated average of the same type as the initial_value.
        """
        ans = 0

        number_of_samples = self.blockSize * self.current_block_index + \
            self.current_index_in_current_block

        if number_of_samples > 0:
            current_average = self.current_block_sum / float(number_of_samples)
            if self.current_block_index > 0:
                current_average = current_average + \
                    self.sum_of_block_averages / \
                    float(self.current_block_index)
                tmp = self.current_index_in_current_block * self.sum_of_block_averages
                tmp = tmp / float(number_of_samples) / \
                    float(self.current_block_index)
                current_average = current_average - tmp
            ans = current_average
        return ans

    def clear(self):
        """
        Clear the accumulated average.
        """
        self.current_block_index = 0
        self.blockSize = 10000
        self.current_index_in_current_block = 0
        self.current_block_sum = self.initial_value
        self.sum_of_block_averages = self.initial_value
