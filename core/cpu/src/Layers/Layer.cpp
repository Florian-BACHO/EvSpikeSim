//
// Created by Florian Bacho on 23/01/23.
//

#include <evspikesim/Layers/Layer.h>

using namespace EvSpikeSim;

Layer::Layer(const std::initializer_list<unsigned int> &weights_dims,
             unsigned int n_inputs,
             unsigned int n_neurons,
             float tau_s,
             float threshold,
             Initializer &initializer,
             unsigned int buffer_size) :
        n_inputs(n_inputs),
        n_neurons(n_neurons),
        tau_s(tau_s),
        tau(2.0f * tau_s),
        threshold(threshold),

        thread_pool(nullptr),
        post_spikes(),
        n_spikes(n_neurons),

        weights(weights_dims, initializer),

        current_pre_spike(n_neurons),
        a(n_neurons),
        b(n_neurons),

        buffer(n_neurons * buffer_size),
        buffer_size(buffer_size),
        buffer_full(false) {}

void Layer::set_thread_pool(std::shared_ptr<EvSpikeSim::ThreadPool> &new_thread_pool) {
    thread_pool = new_thread_pool;
}

void Layer::reset(const SpikeArray &pre_spikes) {
    std::fill(current_pre_spike.begin(), current_pre_spike.end(), pre_spikes.begin());
    std::fill(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 0.0f);
    post_spikes.clear();
    std::fill(n_spikes.begin(), n_spikes.end(), 0);
}

void Layer::reset_buffer() {
    std::fill(buffer.begin(), buffer.end(), infinity);
    buffer_full = false;
}