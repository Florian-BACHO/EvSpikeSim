======
Docker
======

To avoids any troubles that can be encountered when building EvSpikeSim, we provide Dockerfiles for both CPU and GPU
implementations. Both the C++ Library and Python API are installed in the docker images.

EvSpikeSim CPU Docker Image
===========================

To build the docker image of EvSpikeSim for CPU, run from the project root:

.. code-block:: bash

    docker build -t evspikesim_cpu -f docker/cpu/Dockerfile .

EvSpikeSim GPU Docker Image
===========================

.. note::
    `nvidia-docker <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_
    is required to be able to use GPUs in Docker containers.

After installing nvidia-docker, run the following command from the project root
to build the docker image of EvSpikeSim for GPU:


.. code-block:: bash

    nvidia-docker build -t evspikesim_gpu -f docker/gpu/Dockerfile .
