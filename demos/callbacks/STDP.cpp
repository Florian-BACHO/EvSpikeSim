//
// Created by Florian Bacho on 14/02/23.
//

#include <evspikesim/Layers/InferKernel.h>

static constexpr float a_pre = 1.0;
static constexpr float a_post = 1.0;

// Get time constants of each synaptic trace and each neuron trace respectively
extern "C" std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> get_traces_tau(float tau_s, float tau) {
    (void) tau_s;

    // Set tau_pre and tau_post as the membrane time constant.
    return {{tau, INFINITY},
            {tau}};
}

CALLBACK void EvSpikeSim::on_pre_neuron(float weight, float *neuron_traces) {
    (void) weight;
    (void) neuron_traces;
}

CALLBACK void EvSpikeSim::on_pre_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
    (void) weight;
    (void) neuron_traces;

    synaptic_traces[0] += a_pre;
    synaptic_traces[1] -= neuron_traces[0];
}

CALLBACK void EvSpikeSim::on_post_neuron(float *neuron_traces) {
    neuron_traces[0] += a_post;
}

CALLBACK void EvSpikeSim::on_post_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
    (void) weight;
    (void) neuron_traces;

    synaptic_traces[1] += synaptic_traces[0];
}