//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <limits>
#include <initializer_list>
#include <evspikesim/SpikeArray.h>
#include <evspikesim/Layers/LayerDescriptor.h>
#include <evspikesim/Misc/ThreadPool.h>
#include <evspikesim/Misc/NDArray.h>

namespace EvSpikeSim {
    class Layer {
    public:
        Layer(const LayerDescriptor &desc, std::shared_ptr<ThreadPool> &thread_pool,
              const std::initializer_list<unsigned int> &weights_dims, unsigned int buffer_size);

        Layer(const LayerDescriptor &desc, std::shared_ptr<ThreadPool> &thread_pool,
              const std::initializer_list<unsigned int> &weights_dims, Initializer &initializer,
              unsigned int buffer_size);

        virtual ~Layer() = default;

        virtual const SpikeArray &infer(const SpikeArray &pre_spikes) = 0;

        inline const LayerDescriptor &get_descriptor() const { return desc; };

        inline auto &get_weights() { return weights; };

        inline const SpikeArray &get_post_spikes() const { return post_spikes; };

        inline const auto &get_n_spikes() const { return n_spikes; }

    protected:
        void reset(const SpikeArray &pre_spikes);
        void reset_buffer();
        void process_buffer();

    protected:
        static constexpr float infinity = std::numeric_limits<float>::infinity();

        const LayerDescriptor desc;
        SpikeArray post_spikes;
        std::shared_ptr<ThreadPool> thread_pool;
        std::vector<unsigned int> n_spikes; // Counts number of post spikes per neuron
        NDArray<float> weights;
        std::vector<SpikeArray::const_iterator> current_pre_spike; // Keeps track of pre spikes during inference
        std::vector<float> a; // Sum w * exp(t/tau_s)
        std::vector<float> b; // Sum w * exp(t/tau) - reset
        std::vector<float> buffer; // Buffer for spike times
        unsigned int buffer_size;
        bool buffer_full; // Set to true when a neuron's buffer is full
    };
}