//
// Created by Florian Bacho on 14/02/23.
//

#include <evspikesim/Layers/InferKernel.h>

extern "C" unsigned int n_synaptic_traces = 0u; // Number of traces per synapse
extern "C" unsigned int n_neuron_traces = 0u; // Number of traces per neuron

CALLBACK float EvSpikeSim::on_pre(const Spike &pre_spike, float weight) {
    (void)pre_spike;

    return 2.0f * weight; // Weight scaling
}