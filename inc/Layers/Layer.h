//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <limits>
#include <initializer_list>
#include "Misc/ThreadPool.h"
#include "SpikeArray.h"
#include "Layers/LayerDescriptor.h"
#include "Misc/NDArray.h"

namespace EvSpikeSim {
    class Layer {
    public:
        template <typename... Args>
        Layer(const LayerDescriptor &desc, std::shared_ptr<ThreadPool> &thread_pool,
              const std::initializer_list<unsigned int> &weights_dims, Args... args);
        virtual ~Layer() = default;

        virtual const SpikeArray &infer(const SpikeArray &pre_spikes) = 0;

        inline const LayerDescriptor &get_descriptor() const { return desc; };
        inline NDArray<float> &get_weights() { return weights; };
        inline const SpikeArray &get_post_spikes() const { return post_spikes; };
        inline const std::vector<unsigned int> &get_n_spikes() const { return n_spikes; }

    protected:
        void reset();
        void reset_buffer();

    protected:
        static constexpr float infinity = std::numeric_limits<float>::infinity();
        static constexpr unsigned int buffer_size = 64u;

        const LayerDescriptor desc;
        SpikeArray post_spikes;
        std::vector<unsigned int> n_spikes;
        NDArray<float> weights;

        std::shared_ptr<ThreadPool> thread_pool;
        std::vector<float> a;
        std::vector<float> b;
        std::vector<float> buffer;
        bool buffer_full;
    };

    template <typename... Args>
    Layer::Layer(const LayerDescriptor &desc, std::shared_ptr<ThreadPool> &thread_pool,
                 const std::initializer_list<unsigned int> &weights_dims, Args... args) :
            desc(desc), post_spikes(), n_spikes(desc.n_neurons), weights(weights_dims, args...),
            thread_pool(thread_pool), a(desc.n_neurons), b(desc.n_neurons), buffer(desc.n_neurons * buffer_size),
            buffer_full(false) {}
}