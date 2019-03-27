User Guide
==========

.. toctree::
    :glob:
    :caption: Contents:
    :maxdepth: 2

Welcome to the acwind user guide which will take you through the basic
functionallity of the acwind program. If you have any questions on any
of the below please contact the authors. To make sure that you have installed
the package correctly, the following should not generate an error:

.. code:: python3

    import acwind as acw
    print(acw.__version__)

Assuming that the package has been correctly installed we will now
outline how each of the acwind modules work.

The packages has been designed to build neural net classifiers to label SCADA
data coming from wind farms, to perform automatic labelling of operational
states given a set of input signals. For more details see paper_.

Formatting of the Data
----------------------

Data handling in Python is easy using Pandas_. The acwind library does not
assume anything about the data except that features and signals are stored as
floats and that the labelling is stored as either `int` or `bool`. Each column
can contain either a timestamp, float, bool or int. When using the
functionallity in the :ref:`Normalisation module`, acwind will assume that
anything stored as a float is a feature and should thus be normalised over.

**Reminder:** The :ref:`Normalisation module` is implicitly used in
the :ref:`Alignment module`.

.. _Pandas: https://pandas.pydata.org

Creating Multi Farm Training Data
---------------------------------

As we have described in paper_ the predictive power of classifying SCADA data
can be improved by creating multifarm training data. To this end we have
created the :ref:`Alignment module` which generates aligned SCADA data which
can used to train e.g a CNN classifier.

Training a Neural Net Classifier
--------------------------------

We have created a small interface to PyTorch_ which can be used to train
several types of Neural Networks. Please have a look in the source code on
:ref:`Neural Network Model module` to check how this can be used or get in
contact with the Authors. As described in the paper_ these classifiers have an
accuracy of about 95%.

.. _PyTorch: https://pytorch.org
.. _paper: https://arxiv.org/abs/1903.08901
