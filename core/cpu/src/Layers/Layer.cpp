//
// Created by Florian Bacho on 23/01/23.
//

#include <evspikesim/Layers/Layer.h>

using namespace EvSpikeSim;

Layer::Layer(const LayerDescriptor &desc, std::shared_ptr<ThreadPool> &thread_pool,
             const std::initializer_list<unsigned int> &weights_dims, unsigned int buffer_size) :
        desc(desc),
        post_spikes(),
        thread_pool(thread_pool),
        n_spikes(desc.n_neurons),
        weights(weights_dims),
        current_pre_spike(desc.n_neurons),
        a(desc.n_neurons),
        b(desc.n_neurons),
        buffer(desc.n_neurons * buffer_size),
        buffer_size(buffer_size),
        buffer_full(false) {}


Layer::Layer(const LayerDescriptor &desc, std::shared_ptr<ThreadPool> &thread_pool,
             const std::initializer_list<unsigned int> &weights_dims, Initializer &initializer,
             unsigned int buffer_size) :
        desc(desc),
        post_spikes(),
        thread_pool(thread_pool),
        n_spikes(desc.n_neurons),
        weights(weights_dims, initializer),
        current_pre_spike(desc.n_neurons),
        a(desc.n_neurons),
        b(desc.n_neurons),
        buffer(desc.n_neurons * buffer_size),
        buffer_size(buffer_size),
        buffer_full(false) {}

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