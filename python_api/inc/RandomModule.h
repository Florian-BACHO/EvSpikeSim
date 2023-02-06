//
// Created by Florian Bacho on 31/01/23.
//

#pragma once

#include <pybind11/pybind11.h>
#include <evspikesim/Misc/RandomGenerator.h>

extern EvSpikeSim::RandomGenerator global_random_generator;

pybind11::module create_random_module(pybind11::module &parent_module);