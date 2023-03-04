#!/usr/bin/env bash

wget --recursive --level=1 --cut-dirs=3 --no-host-directories --directory-prefix=mnist --accept '*.gz' http://yann.lecun.com/exdb/mnist/
pushd mnist
gunzip *
popd