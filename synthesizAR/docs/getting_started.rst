===============
Getting Started
===============

Dependencies
------------
synthesizAR is compatible with Python 3.6+. The easiest and most convenient way to install Python is through the `Anaconda <https://www.continuum.io/downloads>`_ distribution. Additionally, synthesizAR requires several other packages from the scientific Python ecosystem.

- astropy
- dask
- distributed
- h5py
- fiasco (Install via the `GitHub repository <https://github.com/wtbarnes/fiasco>`_)
- matplotlib
- numpy
- numba
- plasmapy (Install with `pip install plasmapy`)
- scipy
- sunpy
- yt

Unless otherwise noted, all of these packages can be downloaded using the command,

.. code-block:: bash

    $ conda install -c conda-forge {package-name}

See the following section for an easier way to install all of the needed dependencies with one command.

Install
-------
To download synthesizAR, first clone the GitHub repository,

.. code-block:: bash

   $ git clone https://github.com/wtbarnes/synthesizAR.git
   $ cd synthesizAR

The easiest way to grab all of the dependencies is to create a new conda-environment and install them into there. This can be done easily with the included environment file,

.. code-block:: bash

   $ conda env create -f environment.yml
   $ source activate synthesizar

This will create a new `conda environment <http://conda.pydata.org/docs/using/envs.html>`_ with the needed dependencies and activate the environment. Alternatively, you can install all of the packages listed in the above section manually. Finally, to install synthesizAR,

.. code-block:: bash

   $ python setup.py install

Updating
--------
As synthesizAR is not yet available on any package manager, the best way to keep up with releases is to pull down updates from GitHub. To grab the newest version from GitHub and install it, inside the package repo,

.. code-block:: bash

   $ git pull
   $ python setup.py install

If you'd like to maintain a fork of synthesizAR (e.g. if you need to make changes to the codebase or contribute to the package), see the :doc:`Dev page </develop>`.
