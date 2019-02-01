"""
.. module:: alignment

This module contains functions which can be applied to align the baselines of
the SCADA data in some feature set. The module also contains functions which
applies these shifts to Pandas dataframes in either of two choosen features.
**Note:** It is recommended to plot the results of the get_baseline function
before applying the baseline shifts, to make sure that the proposed shift is
sensible.

.. moduleauthor:: amartinsson, ZofiaTr
"""
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import curve_fit, minimize

from .helpers import gaussian


def apply_dataframe_shift(df, features, old_baseline, new_baseline,
                          shift_feature=None):
    """ Function which applies a baseline shift to the dataframe given, where
    after the shift the two baselines will be aligned. If no shift_feature is
    given then the shift is applied to features[0]

    :param pandas df: SCADA data
    :param str features: array of feature names
    :param ndarray old_baseline: baseline of the features in the dataframe
    :param ndarray new_baseline: baseline which the dataframe is supposed to be\
                                 aligned with
    :param str shift_feature: name of the feautre which we want to shift
    """
    # check that we have the right number of featuers
    if len(features) is not 2:
        raise RuntimeError('The number of features must be given as 2')

    # extract the features as a numpy array
    data_to_shift = dataframe[features].values

    if shift_feature is features[0] or shift_feature is None:
        # get the shifted feature
        shifted_feature = shift_data_x(data_to_shift, old_baseline,
                                       new_baseline)

        # assign the shifted feature to the dataframe
        dataframe.loc[:, features[0]] = shifted_feature

    elif shift_feature is features[1]:
        # get the shifted feature
        shifted_feature = shift_data_y(data_to_shift, old_baseline,
                                       new_baseline)

        # assign the shifted feature to the dataframe
        dataframe.loc[:, features[1]] = shifted_feature


def get_baseline(df, features, nbins=200, gauss_fit=False, legacy=False):
    """ 2D histogram approach to get the baseline

    :param pandas df: SCADA data
    :param str features: list of two strings of feature names used for\
                         the histogram.
    :param int nbins: the number of bins in one dimension of the histogram
    :param bool gauss_fit: controls if trying a gaussian fit to smooth the\
                           baseline with
    :param bool legacy: Use the original baseline code instead of the new code

    :return: two arrays with the x, y values of the baseline
    :rtype: ndarray, ndarray
    """
    if (legacy):
        # check that name is element of dataframe
        if np.isin(features[0], dataframe.columns.values)\
                and np.isin(features[1], dataframe.columns.values):

            # histogram the data in nbins * nbins
            histogram, x, y = np.histogram2d(dataframe[features[0]].values,
                                             dataframe[features[1]].values,
                                             bins=nbins, normed=True)

            # get the deltas
            xdelta = 0.5 * (x[1] - x[0])
            ydelta = 0.5 * (y[1] - y[0])

            # make new bins
            xcent = x[0:-1] + xdelta
            ycent = y[0:-1] + ydelta

            # make the mean
            mean = []
            ymean = []

            # loop over the ybins
            for k in range(1, nbins-1):
                # get the histogram and max index
                hist = histogram[:, k]
                index = np.argmax(hist)

                if gauss_fit:
                    try:
                        # get the fitted mean
                        mu = xcent[index]
                        std = np.mean(ycent * (xcent - mu)**2)

                        # fit the curve
                        popt, _ = curve_fit(
                            gaussian, xcent, hist, p0=[1, mu, std])

                        # get the max bin
                        mean = np.append(mean, popt[1])
                        ymean = np.append(ymean, ycent[k])

                    except RuntimeError:
                        raise RuntimeError('Gaussian curve_fit did not '
                                           'converge! Try using paramater '
                                           'option gauss_fit=False instead.')
                        break
                else:
                    # get the max bin
                    mean = np.append(mean, xcent[index])
                    ymean = np.append(ymean, ycent[k])

        # make return array
        return np.array((mean, ymean)).T

    xx = dataframe[features[0]].values
    yy = dataframe[features[1]].values
    aa, bb, cc = np.histogram2d(xx, yy, bins=nbins)
    cc = np.linspace(cc[0], cc[-1], nbins)

    xii = 1
    bx = np.zeros(nbins)
    by = np.zeros(nbins)
    for ii in range(nbins):
        bx[ii] = (bb[ii+1] + bb[ii])/2
        xii = xii+np.argmax(aa[ii, (xii):])
        by[ii] = cc[xii]  # + cc[xii+1])/2

    return np.vstack([bx, by]).T


def shift_data_y(data, old_baseline, new_baseline):
    """ This function returns the shifted values such that the databaselines
    align with each other. The shift is applied in the y direction

    :param ndarray data: data to shift
    :param ndarray old_baseline: values of the current baseline of the data
    :param ndarray new_baseline: values of the new baseline which we want to\
                                 map onto

    :returns: y_shifted, the shifted data in y direction
    :rtype: ndarray
    """
    x_ref = new_baseline[:, 0]
    y_ref = new_baseline[:, 1]

    x_old = old_baseline[:, 0]
    y_old = old_baseline[:, 1]

    # define bins of histogram in x from reference data:
    # using k nearest neighbors with k=1
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(x_ref[:, np.newaxis], np.arange(len(x_ref)))

    # define bins of histogram in y from reference data:
    # using k nearest neighbors with k=1
    neigh_power = KNeighborsClassifier(n_neighbors=1)
    neigh_power.fit(y_old[:, np.newaxis], np.arange(len(y_old)))

    # get y bin index for new data
    index_own = neigh_power.predict(data[:, 1][:, np.newaxis])

    # get x bin index
    index = neigh.predict(x_old[index_own][:, np.newaxis])

    # calculate the shifts
    shifts = y_ref[index] - y_old[index_own]

    return data[:, 1] + shifts


def shift_data_x(data, old_baseline, new_baseline):
    """ This function returns the shifted values such that the databaselines
    align with each other. The shift is applied in the y direction

    :param ndarray data: data to shift
    :param ndarray old_baseline: values of the current baseline of the data
    :param ndarray new_baseline: values of the new baseline which we want to\
                                 map onto

    :returns: x_shifted, the shifted data in x direction
    :rtype: ndarray
    """
    # flip all the directons
    data_tmp = np.zeros(data.shape)
    data_tmp[:, 0], data_tmp[:, 1] = data[:, 1], data[:, 0]

    old_baseline_tmp = np.zeros(old_baseline.shape)
    old_baseline_tmp[:, 0], old_baseline_tmp[:, 1] = old_baseline[:, 1],\
        old_baseline[:, 0]
    new_baseline_tmp = np.zeros(new_baseline.shape)
    new_baseline_tmp[:, 0], new_baseline_tmp[:, 1] = new_baseline[:, 1],\
        new_baseline[:, 0]
    # now apply the shift function
    x_shifted = shift_data_y(data_tmp, old_baseline_tmp, new_baseline_tmp)

    # return the shifted data
    return x_shifted


def normalize_dataset_minmax(df, features=None, classes=None):
    """ Normalizes the dataset uniformly between 0 and 1

    :param pandas df: SCADA data
    :param list features: list of features to be used in the normalization.
    :param classes: list of operational classes to be used in the normalization.

    :return: The normalized dataset
    :rtype: pandas df
    """

    if (features is None):
        raise ValueError('feature list not given!')

    if (len(features) is not 2):
        raise ValuesError('feature length must be of length 2!')

    classes = classes + features

    for ff in flags:
        if ((ff in df.columns) == False):
            df[ff] = 0

    df[features] -= df[features].min()
    df[features] /= df[features].max()

    hh, aa = np.histogram(df[features[1]].values, bins=100)
    aa = (aa[:-1] + aa[1:])/2
    ii = np.argmax(hh[aa < 0.1])
    df[features[1]] -= (aa[ii])
    hh, aa = np.histogram(df[features[1]].values, bins=100)
    aa = (aa[:-1] + aa[1:])/2
    hh = hh[aa > 0.9]
    aa = aa[aa > 0.9]
    ii = np.argmax(hh)
    df[features[1]] /= (aa[ii])

    if 'pitch' in df:
        for ii in range(5):
            df.loc[df['pitch'] < -60,
                   'pitch'] = df.loc[df['pitch'] < -60, 'pitch'] + 360
            df.loc[df['pitch'] > 300,
                   'pitch'] = df.loc[df['pitch'] > 300, 'pitch'] - 360

    return df


def normalize_dataset(df, relativeto=None, bins=100, qstep=100, verbose=False,
                      features=None):
    """ Normalize a dataset, and shift one onto another if relativeto\
    is specified.

    :param pandas df: SCADA data
    :param pandas relativeto: The target dataset to match with. It will\
                                 scale df to fit this dataset.
    :param int bins: the number of bins to use for the histogram
    :param int qstep: How much the data should be subsampled. Larger numbers \
                      give improved performance.
    :param bool verbose: Whether to print to screen or not.
    :param list features: A list of the features to be normalized.

    :return: A dataset normalized and scaled to match the one specified
    :rtype: dataset
    """
    if (features is None):
        raise ValueError('feature list not given!')

    if (len(features) is not 2):
        raise ValuesError('feature length must be of length 2!')

    df = normalize_dataset_minmax(df, features=features)

    if (relativeto is None):
        if verbose:
            print('>) Done!')
        return df
    zz = relativeto[[features[0], features[1]]].values
    H, _, __ = np.histogram2d(zz[:, 0], zz[:, 1], bins=bins, range=[
                              [-.1, 1.1], [-.1, 1.1]], normed=True)

    zz = df[[features[0], features[1]]].values
    zz_q = zz[::qstep, :]
    aa = np.linspace(0.5, 1.5, 10)
    bb = np.linspace(-.2, .2, 10)
    cc = np.copy(bb)
    if verbose:
        print('  o) Scanning over values...')
    minres = np.inf
    for a in aa:
        for b in bb:
            for c in cc:
                res = l1diff([a, b, c], zz_q, H, bins)
                if res < minres:
                    minres = res
                    x0 = np.copy([a, b, c])
    if (verbose):
        print('  o) Beginning minimization step at ' +
              str(x0) + ' with fun=' + str(minres))
    res = minimize(l1diff, x0, args=(zz, H, bins), method='Nelder-Mead')
    a, b, c = res.x
    df[features[0]] *= a
    df[features[0]] += b
    df[features[0]] += c * df[features[1]].values
    if (verbose):
        print('>) Final L1diff: ' + str(res.fun))
        print('   scale: ' + str(a) + ', shift: ' +
              str(b) + ', bend: ' + str(c))

    return df


def l1diff(x, zz, HH, bins):
    """ Computes the L-1 difference between a scaled dataset and a target\
    histogram

    :param list x: Three floats corresponding to scaling factors
    :param array zz: The dataset to be scaled
    :param array HH: The 2d histogram
    :param int bins: the number of bins to use for the histogram

    :return: The approximate L-1 difference
    :rtype: float
    """
    a, b, c = x

    # Compare histogram after scaling and shifting
    H, _, __ = np.histogram2d(a*zz[:, 0]+b+c*zz[:, 1], zz[:, 1],
                              bins=bins, range=[[-.1, 1.1], [-.1, 1.1]],
                              normed=True)

    return np.sum(np.abs(H-HH))
