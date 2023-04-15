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
                 get_traces_tau_fct traces_tau_fct,
                 infer_kernel_fct kernel) :
        Layer(n_inputs, n_neurons, tau_s, threshold, initializer, buffer_size, traces_tau_fct, kernel) {}

