//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <vector>
#include "Layers/Layer.h"
#include "Layers/FCLayer.h"
#include "Misc/ThreadPool.h"
#include "Misc/NDArray.h"

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

        // Iterator
        inline iterator begin() { return layers.begin(); };
        inline iterator end() { return layers.end(); };

        // Accessor
        template <typename T>
        inline std::shared_ptr<Layer> operator[](T idx) { return layers[idx]; }
        inline std::shared_ptr<Layer> get_output_layer() { return layers.back(); }

        inline auto get_n_layers() const { return layers.size(); }

    private:
        std::shared_ptr<ThreadPool> thread_pool;
        std::vector<std::shared_ptr<Layer>> layers;
    };
}