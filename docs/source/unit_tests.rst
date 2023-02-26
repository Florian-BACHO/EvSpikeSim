==========
Unit Tests
==========

C++ Library Unit Tests
======================

Dependencies
------------

Unit tests of the C++ depends on the following package:

- googletest >= 1.13.0

which can be installed as follows:

.. code-block:: bash

    sudo apt-get install libgtest-dev

Tests
-----

Run CMake **without** the ``-DNO_TEST=ON`` argument:

.. code-block:: bash

    cd build
    cmake ../core

You can now build and run the tests:

.. code-block:: bash

    make tests
    ./tests/tests

.. note::
    This only tests EvSpikeSim for the processing unit (CPU or GPU) targeted by the build.

Python API Unit Tests
=====================

After installing the EvSpikeSim Python package on your system, run the following command to run the unit tests:

.. code-block:: bash

    cd python_api/tests
    python3 -m unittest discover -p "*_test.py"

.. note::
    This only tests EvSpikeSim for the processing unit (CPU or GPU) targeted by the build.