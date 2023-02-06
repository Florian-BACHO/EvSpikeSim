//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <evspikesim/Layers/FCLayerDescriptor.h>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Misc/NDArray.h>
#include <evspikesim/Misc/ThreadPool.h>

namespace EvSpikeSim {
    class FCLayer : public Layer {
    public:
        using descriptor_type = FCLayerDescriptor;

    public:
        template <typename... Args>
        FCLayer(const descriptor_type &desc, std::shared_ptr<ThreadPool> thread_pool, Args... args) :
                Layer(desc, thread_pool, {desc.n_neurons, desc.n_inputs}, args...) {}

        const SpikeArray &infer(const SpikeArray &pre_spikes) override;

    private:
        void apply_decay_and_weight(unsigned int neuron_idx, float delta_t, unsigned int pre_idx);
        void fire(unsigned int neuron_idx, float current_time, float next_pre_time, unsigned int &n_spike_buffer);
        void infer_neuron(const SpikeArray &pre_spikes, unsigned int neuron_start);
        void infer_range(const SpikeArray &pre_spikes, unsigned int neuron_start, unsigned int neuron_end);
        void process_buffer();
    };
}