//
// Created by Florian Bacho on 16/02/23.
//

#include <evspikesim/Layers/InferKernel.h>

// Get time constants of each synaptic trace and each neuron trace respectively
extern "C" std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> get_traces_tau(float tau_s, float tau) {
    (void) tau_s;
    (void) tau;

    return {{},
            {}}; // No trace
}

CALLBACK void EvSpikeSim::on_pre_neuron(float weight, float *neuron_traces) {
    (void) weight;
    (void) neuron_traces;
}

CALLBACK void EvSpikeSim::on_pre_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
    (void) weight;
    (void) neuron_traces;
    (void) synaptic_traces;
}

CALLBACK void EvSpikeSim::on_post_neuron(float *neuron_traces) {
    (void) neuron_traces;
}

CALLBACK void EvSpikeSim::on_post_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
    (void) weight;
    (void) neuron_traces;
    (void) synaptic_traces;
}