//
// Created by Florian Bacho on 16/02/23.
//

#include <evspikesim/Layers/EligibilityTraces.h>

EvSpikeSim::vector<float> EvSpikeSim::synaptic_traces_tau(float tau_s, float tau) {
    (void) tau_s;
    (void) tau;

    return {};
}

EvSpikeSim::vector<float> EvSpikeSim::neuron_traces_tau(float tau) {
    (void) tau;

    return {};
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