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
        FCLayer(const descriptor_type &desc, std::shared_ptr<ThreadPool> &thread_pool, unsigned int buffer_size = 64u);

        FCLayer(const descriptor_type &desc, std::shared_ptr<ThreadPool> &thread_pool, Initializer &initializer,
                unsigned int buffer_size = 64u);

        FCLayer(const descriptor_type &desc, std::shared_ptr<ThreadPool> &thread_pool, Initializer &&initializer,
                unsigned int buffer_size = 64u);

        const SpikeArray &infer(const SpikeArray &pre_spikes) override;

    private:
        void apply_decay_and_weight(unsigned int neuron_idx, float delta_t, unsigned int pre_idx);

        bool fire(unsigned int neuron_idx, float current_time, float next_pre_time, unsigned int &n_spike_buffer);

        void infer_neuron(const SpikeArray &pre_spikes, unsigned int neuron_start, bool first_call);

        void infer_range(const SpikeArray &pre_spikes, unsigned int neuron_start, unsigned int neuron_end,
                         bool first_call);
    };
}