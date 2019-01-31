acwind
======
A Python library for automatic classification of SCADA wind energy analytics

License: GPL-3.0 (see the LICENSE for details)

Download
--------

You can install this library using Anaconda,
https://www.anaconda.com/download

To install the anaconda version

..code:: none
    $ conda install -c acwind-lib acwind

To install the git version

..code:: none
    $ git clone git@github.com:acwind-lib/acwind.git

Documentation
-------------

The docs can be found at:
https://acwind-lib.readthedocs.io/en/latest/

You can also generate the documentation locally by::
    $ cd docs
    $ make html

which will generate all the documentation in `_build/html`

Installation
------------

The package has the following dependencies

    * itertools
    * math
    * numpy
    * pandas
    * Python 3
    * pytorch
    * scipy
    * sklearn

To install the package simply run

..code:: none
    $ python setup.py install

Development
-----------

In case you want to contribute to this project contact the authors by now.

Acknowledgements
----------------

This project was done under supervision of Prof Ben Leimkuhler and in
collaboration with Michael Wilkinson, Jon Collins, Thomas van Delft and
Sergio Jimenez Sanjuan. The authors thank to EPSRC IAA grant number PIII015 and
DNV GL for funding this project.
