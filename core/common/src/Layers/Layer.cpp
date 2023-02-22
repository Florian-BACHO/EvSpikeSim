//
// Created by Florian Bacho on 08/02/23.
//

#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Misc/Algorithms.h>

using namespace EvSpikeSim;

Layer::Layer(const std::initializer_list<unsigned int> &weights_dims,
             unsigned int n_inputs,
             unsigned int n_neurons,
             float tau_s,
             float threshold,
             Initializer &initializer,
             unsigned int buffer_size,
             get_traces_tau_fct traces_tau_fct,
             infer_kernel_fct kernel_fct) :
        n_inputs(n_inputs),
        n_neurons(n_neurons),
        tau_s(tau_s),
        tau(2.0f * tau_s),
        threshold(threshold),

        post_spikes(),
        n_spikes(n_neurons),

        weights(weights_dims, initializer),

        current_pre_spike(n_neurons),
        current_time(n_neurons),
        a(n_neurons),
        b(n_neurons),

        buffer(n_neurons * buffer_size),
        buffer_size(buffer_size),
        buffer_full(EvSpikeSim::make_unique<bool>()),

        kernel_fct(kernel_fct) {
    init_traces(traces_tau_fct);
    kernel_data = get_kernel_data();
}

void Layer::init_traces(get_traces_tau_fct traces_tau_fct) {
    auto traces_tau = traces_tau_fct(tau_s, tau);

    synaptic_traces_tau = traces_tau.first;
    neuron_traces_tau = traces_tau.second;
    synaptic_traces = EvSpikeSim::vector<float>(weights.size() * synaptic_traces_tau.size());
    neuron_traces = EvSpikeSim::vector<float>(n_neurons * neuron_traces_tau.size());
}

KernelData Layer::get_kernel_data() {
    return {
            n_inputs,
            n_neurons,
            tau_s,
            tau,
            threshold,

            n_spikes.data(),

            weights.get_c_ptr(),

            current_pre_spike.data(),
            current_time.data(),
            a.data(),
            b.data(),

            buffer.data(),
            buffer_full.get(),
            buffer_size,

            synaptic_traces_tau.data(),
            neuron_traces_tau.data(),

            synaptic_traces.data(),
            neuron_traces.data(),

            (unsigned int)synaptic_traces_tau.size(),
            (unsigned int)neuron_traces_tau.size()
    };
}

void Layer::process_buffer() {
    float time;

    for (unsigned int i = 0; i < n_neurons; i++) {
        for (unsigned int j = 0; j < buffer_size; j++) {
            time = buffer[i * buffer_size + j];
            if (time == infinity)
                break;
            post_spikes.add(i, time);
        }
    }
}

void Layer::reset(const SpikeArray &pre_spikes) {
    post_spikes.clear();
    EvSpikeSim::fill(n_spikes.begin(), n_spikes.end(), 0u);
    EvSpikeSim::fill(current_pre_spike.begin(), current_pre_spike.end(), &(*pre_spikes.begin()));
    EvSpikeSim::fill(a.begin(), a.end(), 0.0f);
    EvSpikeSim::fill(b.begin(), b.end(), 0.0f);
    EvSpikeSim::fill(current_time.begin(), current_time.end(), 0.0f);
    EvSpikeSim::fill(synaptic_traces.begin(), synaptic_traces.end(), 0.0f);
    EvSpikeSim::fill(neuron_traces.begin(), neuron_traces.end(), 0.0f);
}

void Layer::reset_buffer() {
    EvSpikeSim::fill(buffer.begin(), buffer.end(), infinity);
    *buffer_full = false;
}

const SpikeArray &Layer::infer(const SpikeArray &pre_spikes) {
    bool first_call = true;
    auto *end_pre_ptr = &(*(pre_spikes.end()));
    auto *thread_pool_ptr = get_thread_pool_ptr();

    // Reset layer
    reset(pre_spikes);

    // Check if pre_spikes
    if (pre_spikes.is_empty())
        return post_spikes;
    // Infer
    do {
        reset_buffer();
        kernel_fct(kernel_data, end_pre_ptr, first_call, thread_pool_ptr);
        process_buffer();
        first_call = false;
    } while (*buffer_full);

    // Sort and return post-synaptic spikes
    post_spikes.sort();
    return post_spikes;
}