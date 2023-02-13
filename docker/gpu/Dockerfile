FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Run from project root: nvidia-docker build -t evspikesim_gpu -f docker/gpu/Dockerfile .

WORKDIR /usr/evspikesim

COPY . .

RUN apt-get update

# C++ Library

#   Install dependencies
RUN apt-get install -y g++ cmake

#   Build and install C++ EvSpkeSim library
RUN rm -rf build
RUN mkdir -p build
RUN cd build && \
    cmake -DBUILD_GPU=ON -DNO_TEST=ON ../core && \
    make evspikesim -j && \
    make install

# Python3 API

#   Install dependencies
RUN rm -rf python_api/build
RUN apt-get install -y python3-dev python3-pip
RUN cd python_api && \
    pip3 install -r requirements.txt
RUN cd python_api && \
    python3 setup.py install --gpu