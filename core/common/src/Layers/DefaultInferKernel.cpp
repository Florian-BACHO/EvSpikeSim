//
// Created by Florian Bacho on 16/02/23.
//

#include <evspikesim/Layers/InferKernel.h>

extern "C" unsigned int n_synaptic_traces = 0u;
extern "C" unsigned int n_neuron_traces = 0u;

CALLBACK float EvSpikeSim::on_pre(const EvSpikeSim::Spike &pre_spike, float weight) {
    (void)pre_spike;

    return weight;
}