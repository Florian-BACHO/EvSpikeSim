//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <vector>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/Misc/JITCompiler.h>

namespace EvSpikeSim {
    class SpikingNetwork {
    public:
        using iterator = std::vector<std::shared_ptr<Layer>>::iterator;

        static constexpr char default_compile_path[] = "/tmp/evspikesim";

    public:
        SpikingNetwork(const std::string &compile_path = default_compile_path);

        ~SpikingNetwork();

        template<class LayerType, typename... Args>
        std::shared_ptr<LayerType> add_layer(Args... args) {
            auto layer = std::make_shared<LayerType>(args...);

            layers.push_back(layer);
            return layer;
        }

        template<class BaseLayerType, typename... Args>
        std::shared_ptr<BaseLayerType> add_layer_from_source(const std::string &src_path, Args... args) {
            // Compile source file
            auto &dlib = (*compiler)(src_path);

            // Load extern "C" kernel
            auto kernel_fct = reinterpret_cast<typename BaseLayerType::kernel_signature>
            (dlib(BaseLayerType::kernel_symbol));

            // Create layer
            auto layer = std::make_shared<BaseLayerType>(args..., kernel_fct);

            layers.push_back(layer);
            return layer;
        }

        const SpikeArray &infer(const SpikeArray &pre_spikes);

        template<class IndicesType, class TimesType>
        const SpikeArray &infer(const IndicesType &indices, const TimesType &times) {
            SpikeArray input_spikes(indices, times);

            input_spikes.sort();
            return infer(input_spikes);
        }

        // Iterator
        iterator begin();

        iterator end();

        // Accessor
        template<typename T>
        std::shared_ptr<Layer> operator[](T idx) { return layers[idx]; }

        std::shared_ptr<Layer> get_output_layer();

        unsigned int get_n_layers() const;

    private:
        std::vector<std::shared_ptr<Layer>> layers;
        std::unique_ptr<JITCompiler> compiler;
    };
}