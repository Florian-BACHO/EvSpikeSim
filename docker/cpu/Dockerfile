FROM ubuntu:22.04

# Run from project root: docker build -t evspikesim -f docker/cpu/Dockerfile .

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
    cmake -DNO_TEST=ON ../core && \
    make evspikesim -j && \
    make install

# Python3 API

#   Install dependencies
RUN rm -rf python_api/build
RUN apt-get install -y python3-dev python3-pip
RUN cd python_api && \
    pip3 install -r requirements.txt
RUN cd python_api && \
    python3 setup.py install