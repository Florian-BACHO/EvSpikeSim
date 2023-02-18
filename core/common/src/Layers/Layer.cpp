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
        a(n_neurons),
        b(n_neurons),

        buffer(n_neurons * buffer_size),
        buffer_size(buffer_size),
        buffer_full(EvSpikeSim::make_unique<bool>()),

        kernel_data(get_kernel_data()),
        kernel_fct(kernel_fct) {}

KernelData Layer::get_kernel_data() {
    return {
            n_inputs,
            n_neurons,
            tau_s,
            tau,
            threshold,

            current_pre_spike.data(),
            n_spikes.data(),

            weights.get_c_ptr(),
            a.data(),
            b.data(),

            buffer.data(),
            buffer_full.get(),
            buffer_size
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
    EvSpikeSim::fill(current_pre_spike.begin(), current_pre_spike.end(), &(*pre_spikes.begin()));
    EvSpikeSim::fill(a.begin(), a.end(), 0.0f);
    EvSpikeSim::fill(b.begin(), b.end(), 0.0f);
    post_spikes.clear();
    EvSpikeSim::fill(n_spikes.begin(), n_spikes.end(), 0u);
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