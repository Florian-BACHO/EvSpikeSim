============
Installation
============

.. contents:: Table of Contents

.. note::
    Because EvSpikeSim is still in active development, no official release is available yet.
    However, the simulator can be built and installed from sources.

The following describes step by step how to download, build and install the project on your system.

Download Sources
================

To download the EvSpikeSim sources, clone the project git repository

.. code-block:: bash

    git clone https://github.com/Florian-BACHO/EvSpikeSim.git

Two builds are available: one for CPUs only and one for NVIDIA GPUs.

Dependencies
============

C++ Library Dependencies
------------------------

Both C++ and Python builds of EvSpikeSim require the following packages to be installed:

- g++ >= 9
- cmake >= 3.14

Install these dependencies by running:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install g++ cmake

Python API Dependencies
-----------------------

The Python version of EvSpikeSim has additional dependencies:

- python >= 3.7
- numpy >= 1.18.0
- pybind11 >= 2.10.3

If not already present on your system, Python3 can be installed as follows:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install python3.x

Replace x by the desired version of python (e.g. python3.9).

Required Python packages can either manually by running:

.. code-block:: bash

    pip3 install numpy pybind11[global]

or using the ``requirements.txt`` file located in the ``python_api/`` directory:

.. code-block:: bash

    cd python_api
    pip3 install -r requirements.txt

GPU Dependencies
----------------

The GPU version of EvSpikeSim also requires the following dependency:

- NVIDIA CUDA Toolkit >= 11

Please, see the `installation guide <https://developer.nvidia.com/cuda-downloads>`_ of the NVIDIA CUDA Toolkit SDK.

CPU Install
===========

The following describes how to build and install EvSpikeSim for CPU only.

Build and install C++ library
-----------------------------

To compile the C++ library for CPU, run the following commands from the project root:

.. code-block:: bash

    mkdir build
    cd build
    cmake ../core -DNO_TEST=ON
    make

After compilation, install the library by running:

.. code-block:: bash

    sudo make install

Build and install Python API
----------------------------

.. note::

    The EvSpikeSim C++ Library needs to be installed before building the Python API (see previous section).

To build and install the PythonAPI, run from the project root:

.. code-block:: bash

    cd python_api
    python3 setup.py install

GPU Install
===========

The following describes how to build and install EvSpikeSim for GPUs.
The steps are similar to the CPU installation but additional arguments and environment variables need to be set.

Build and install C++ library
-----------------------------

To compile the C++ library for NVIDIA GPUs, run the following commands from the project root:

.. code-block:: bash

    mkdir build
    cd build
    cmake ../core -DNO_TEST=ON -DBUILD_GPU=ON
    make

In some cases, the compute capability of the target GPU (see `this link <https://developer.nvidia.com/cuda-gpus>`_ to find out which compute capability corresponds to your GPU)
needs to be provided to cmake:

.. code-block:: bash

    cmake ../core/ -DBUILD_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=XX

Here, XX has to be replaced by the corresponding compute capability, e.g.:

.. code-block:: bash

    cmake ../core/ -DBUILD_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=86

for a GPU with a compute capability of 8.6.

Finally, to install the library, run:

.. code-block:: bash

    sudo make install

Build and install Python API
----------------------------

.. note::

    The EvSpikeSim C++ Library needs to be installed before building the Python API (see previous section).

To build and install the PythonAPI, run from the project root:

.. code-block:: bash

    cd python_api
    pip3 install -r requirements.txt
    python3 setup.py install --gpu

If the nvcc compiler is not found, try specifying the path to the Cuda home directory in your environment:

.. code-block:: bash

    CUDAHOME=/path/to/cuda/ python3 setup.py install --gpu