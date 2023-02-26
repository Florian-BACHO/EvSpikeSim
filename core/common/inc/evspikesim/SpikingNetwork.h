//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <vector>
#include <evspikesim/Initializers/ConstantInitializer.h>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/Misc/NDArray.h>
#include <evspikesim/Misc/JITCompiler.h>

namespace EvSpikeSim {
    /**
     * Spiking Neural Network (SNN) composed of layers of spiking neurons.
     * Layers have to be successively added from input to outputs.
     * SpikingNetwork uses the JITCompiler to compile and load custom source files containing kernel callbacks.
     */
    class SpikingNetwork {
    public:
        using iterator = std::vector<std::shared_ptr<Layer>>::iterator; /**< Type definition of the Layer iterator */

        static constexpr char default_compile_path[] = "/tmp/evspikesim"; /**< Default compilation path for custom kernel callback sources. */

    public:
        /**
         * Constructs an empty SNN.
         * @param compile_path Compilation path for custom kernel callback sources.
         */
        SpikingNetwork(const std::string &compile_path = default_compile_path);

        /**
         * Destructor. Deletes layer, unload custom kernel callback dynamic libraries and delete JITCompiler.
         */
        ~SpikingNetwork();

        /**
         * Adds a layer of type LayerType with the given arguments to the network.
         * The added layer uses the default kernel during inference.
         * @tparam LayerType Type of layer to add. Must inherit from Layer.
         * @tparam Args Types of arguments passed to the new layer constructor.
         * @param args Arguments passed to the new layer constructor.
         * @return Shared pointer on the new layer.
         */
        template<class LayerType, typename... Args>
        std::shared_ptr<LayerType> add_layer(Args... args) {
            auto layer = std::make_shared<LayerType>(args...);

            layers.push_back(layer);
            return layer;
        }

        /**
         * Adds a layer of type LayerType with a custom kernel callbacks source.
         * If not already used, the given source file is compiled and loaded by the JITCompiler.
         * @tparam LayerType Type of layer to add. Must inherit from Layer.
         * @tparam Args Types of arguments passed to the new layer constructor.
         * @param src_path Path to the kernel callbacks source file.
         * @param args Arguments passed to the new layer constructor.
         * @return Shared pointer on the new layer.
         * @throw std::runtime_error if the given source file failed to compile.
         * @throw std::runtime_error with the result of dlerror() if the compiled dynamic library could not be loaded.
         */
        template<class LayerType, typename... Args>
        std::shared_ptr<LayerType> add_layer_from_source(const std::string &src_path, Args... args) {
            // Compile source file
            auto &dlib = (*compiler)(src_path);

            // Load extern "C" kernel
            auto kernel_fct = reinterpret_cast<infer_kernel_fct>(dlib(infer_kernel_symbol));
            auto traces_tau_fct = reinterpret_cast<get_traces_tau_fct>(dlib(get_traces_tau_symbol));

            // Create layer
            auto layer = std::make_shared<LayerType>(args..., traces_tau_fct, kernel_fct);

            layers.push_back(layer);
            return layer;
        }

        /**
         * Infer the entire SNN using the given sorteed input spike array.
         * @param pre_spikes Sorted input spikes.
         * @return Constant reference on the output spike array.
         * @throw std::runtime_error if pre_spikes are not sorted.
         */
        const SpikeArray &infer(const SpikeArray &pre_spikes);

        /**
         * Infer the entire SNN using the given input spike indices and times.
         * @tparam IndicesType Type of spike indices
         * @tparam TimesType Type of spike times
         * @param indices Input spike indices.
         * @param times Input spike times.
         * @return Constant reference on the output spike array.
         */
        template<class IndicesType, class TimesType>
        const SpikeArray &infer(const IndicesType &indices, const TimesType &times) {
            SpikeArray input_spikes(indices, times);

            input_spikes.sort();
            return infer(input_spikes);
        }

        // Iterators
        /**
         * Gets an iterator on the first layer.
         * @return An iterator on the first layer.
         */
        iterator begin();

        /**
         * Gets the end iterator of layers.
         * @return The end iterator of layers.
         */
        iterator end();

        // Accessor
        /**
         * Gets the layer at the specified index.
         * @tparam T Type of index.
         * @param idx The index of the layer.
         * @return A shared pointer on the requested layer.
         */
        template<typename T>
        std::shared_ptr<Layer> operator[](T idx) { return layers[idx]; }

        /**
         * Gets the output layer.
         * @return A shared pointer on the last layer.
         */
        std::shared_ptr<Layer> get_output_layer();

        /**
         * Gets the number of layers.
         * @return The number of layers.
         */
        unsigned int get_n_layers() const;

    private:
        std::vector<std::shared_ptr<Layer>> layers; /**< Vector storing the layers */
        std::unique_ptr<JITCompiler> compiler; /**< The JIT compiler used to compile and load custom kernel callback sources */
    };
}