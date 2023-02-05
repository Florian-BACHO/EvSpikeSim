//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <vector>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/Misc/ThreadPool.h>
#include <evspikesim/Misc/NDArray.h>

namespace EvSpikeSim {
    class SpikingNetwork {
    public:
        using iterator = std::vector<std::shared_ptr<Layer>>::iterator;

    public:
        SpikingNetwork() : thread_pool(std::make_shared<ThreadPool>()), layers() {}

        template <typename... Args>
        std::shared_ptr<FCLayer> add_layer(const FCLayerDescriptor &descriptor, Args... args) {
            auto layer = std::make_shared<FCLayer>(descriptor, thread_pool, args...);

            layers.push_back(layer);
            return layer;
        }

        const SpikeArray &infer(const SpikeArray &pre_spikes);

        template <class IndicesType, class TimesType>
        const SpikeArray &infer(const IndicesType &indices, const TimesType &times) {
            SpikeArray input_spikes(indices, times);

            input_spikes.sort();
            return infer(input_spikes);
        }

        // Iterator
        iterator begin();
        iterator end();

        // Accessor
        template <typename T>
        std::shared_ptr<Layer> operator[](T idx) { return layers[idx]; }

        std::shared_ptr<Layer> get_output_layer();

        unsigned int get_n_layers() const;

    private:
        std::shared_ptr<ThreadPool> thread_pool;
        std::vector<std::shared_ptr<Layer>> layers;
    };
}