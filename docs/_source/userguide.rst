User Guide
==========

.. toctree::
    :glob:
    :caption: Contents:
    :maxdepth: 2

Welcome to the acwind user guide which will take you through the basic
functionallity of the acwind program. To make sure that you have installed the
package correctly, the following should not generate an error:

.. code:: python3

    import acwind as acw
    print(acw.__version__)

Assuming that the package has been correctly installed we will now
outline how each of the acwind modules work.

The packages has been designed to build neural net classifiers to label SCADA
data coming from wind farms, to perform automatic labelling of operational
states given a set of input signals. For more details see
''REFERENCE TO PAPER''.

to be continued ...

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

Describe how to use the :ref:`Alignment module` to create a training dataset which
can used to train CNN's

Training a Neural Net Classifier
--------------------------------

Describe how to use :ref:`Neural Network Model module`
