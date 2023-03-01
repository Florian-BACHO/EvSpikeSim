//
// Created by Florian Bacho on 14/02/23.
//

#include <evspikesim/Layers/InferKernel.h>

static constexpr float a_pre = 1e-3;
static constexpr float a_post = 1e-3;

// Get time constants of each synaptic trace and each neuron trace respectively
extern "C" std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> get_traces_tau(float tau_s, float tau) {
    (void) tau_s;

    // Set tau_pre and tau_post as the membrane time constant.
    return {{tau}, {tau}};
}

CALLBACK float EvSpikeSim::on_pre(const EvSpikeSim::Spike &pre_spike, float &weight, float *neuron_traces,
                                  float *synaptic_traces, unsigned int n_synapses) {
    (void) pre_spike;
    (void) n_synapses;

    // Update synaptic trace
    synaptic_traces[0] += a_pre;
    weight -= neuron_traces[0]; // Depression
    return weight;
}

CALLBACK void EvSpikeSim::on_post(float *neuron_weights, float *neuron_traces, float *synaptic_traces,
                                  unsigned int n_synapses) {
    neuron_traces[0] += a_post;
    for (auto i = 0u; i < n_synapses; i++)
        neuron_weights[i] += synaptic_traces[i]; // Potentiation
}