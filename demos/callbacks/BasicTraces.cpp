//
// Created by Florian Bacho on 14/02/23.
//

#include <evspikesim/Layers/InferKernel.h>

static constexpr auto n_synaptic_traces = 2u;

// Get time constants of each synaptic trace and each neuron trace respectively
extern "C" std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> get_traces_tau(float tau_s, float tau) {
    (void) tau_s;

    // Traces with infinite time constant have no leak. They are used to accumulated values
    return {{tau, INFINITY}, // Synaptic traces time constants
            {}}; // Neuron traces time constants
}

CALLBACK float EvSpikeSim::on_pre(const EvSpikeSim::Spike &pre_spike, float &weight, float *neuron_traces,
                                  float *synaptic_traces, unsigned int n_synapses) {
    (void) pre_spike;
    (void) neuron_traces;
    (void) n_synapses;

    // Update synaptic trace
    synaptic_traces[0] += 1.0f;
    return weight;
}

CALLBACK void EvSpikeSim::on_post(float *neuron_weights, float *neuron_traces, float *synaptic_traces,
                                  unsigned int n_synapses) {
    (void) neuron_weights;
    (void) neuron_traces;

    // Accumulates synaptic traces at post-synaptic spikes
    for (auto i = 0u; i < n_synapses; i++)
        synaptic_traces[i * n_synaptic_traces + 1] += synaptic_traces[i * n_synaptic_traces];
}