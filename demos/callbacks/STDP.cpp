//
// Created by Florian Bacho on 14/02/23.
//

#include <evspikesim/Layers/EligibilityTraces.h>

namespace sim = EvSpikeSim;

static constexpr float a_pre = 1.0;
static constexpr float a_post = 1.0;

sim::vector<float> sim::synaptic_traces_tau(float tau_s, float tau) {
    (void) tau_s;

    return {tau, INFINITY};
}

sim::vector<float> sim::neuron_traces_tau(float tau) {
    (void) tau;

    return {tau};
}

CALLBACK void sim::on_pre_neuron(float weight, float *neuron_traces) {
    (void) weight;
    (void) neuron_traces;
}

CALLBACK void sim::on_pre_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
    (void) weight;
    (void) neuron_traces;

    synaptic_traces[0] += a_pre;
    synaptic_traces[1] -= neuron_traces[0];
}

CALLBACK void sim::on_post_neuron(float *neuron_traces) {
    neuron_traces[0] += a_post;
}

CALLBACK void sim::on_post_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
    (void) weight;
    (void) neuron_traces;

    synaptic_traces[1] += synaptic_traces[0];
}