//
// Created by Florian Bacho on 22/01/23.
//

#include <cstdio>
#include <cmath>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/Spike.h>

using namespace EvSpikeSim;

FCLayer::FCLayer(unsigned int n_inputs,
                 unsigned int n_neurons,
                 float tau_s,
                 float threshold,
                 Initializer &initializer,
                 unsigned int buffer_size,
                 infer_kernel_fct kernel) :
        Layer({n_neurons, n_inputs}, n_inputs, n_neurons, tau_s, threshold, initializer, buffer_size, kernel) {}