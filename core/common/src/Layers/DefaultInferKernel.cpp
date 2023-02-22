//
// Created by Florian Bacho on 16/02/23.
//

#include <evspikesim/Layers/InferKernel.h>

// Get time constants of each synaptic trace and each neuron trace respectively
extern "C" std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> get_traces_tau(float tau_s, float tau) {
    (void) tau_s;
    (void) tau;

    return {{}, {}}; // No trace
}

CALLBACK float EvSpikeSim::on_pre(const EvSpikeSim::Spike &pre_spike, float weight, float *neuron_traces,
                                  float *synaptic_traces, unsigned int n_synapses) {
    (void) pre_spike;
    (void) neuron_traces;
    (void) synaptic_traces;
    (void) n_synapses;

    return weight;
}

CALLBACK void EvSpikeSim::on_post(float *neuron_traces, float *synaptic_traces, unsigned int n_synapses) {
    (void) neuron_traces;
    (void) synaptic_traces;
    (void) n_synapses;
}