//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <limits>
#include <initializer_list>
#include <evspikesim/SpikeArray.h>
#include <evspikesim/Layers/InferKernelDeclarations.h>
#include <evspikesim/Misc/NDArray.h>
#include <evspikesim/Misc/ContainerTypes.h>

namespace EvSpikeSim {
    class Layer {
    public:
        Layer(const std::initializer_list<unsigned int> &weights_dims,
              unsigned int n_inputs,
              unsigned int n_neurons,
              float tau_s,
              float threshold,
              Initializer &initializer,
              unsigned int buffer_size,
              infer_kernel_fct kernel_fct = infer_kernel);

        const SpikeArray &infer(const SpikeArray &pre_spikes);

        inline auto get_n_inputs() const { return n_inputs; }

        inline auto get_n_neurons() const { return n_neurons; }

        inline auto get_tau_s() const { return tau_s; }

        inline auto get_tau() const { return tau; }

        inline auto get_threshold() const { return threshold; }

        inline auto &get_weights() { return weights; };

        inline const SpikeArray &get_post_spikes() const { return post_spikes; };

        inline const auto &get_n_spikes() const { return n_spikes; }

    protected:
        void reset(const SpikeArray &pre_spikes);

        void reset_buffer();

        void process_buffer();

    private:
        KernelData get_kernel_data();

        void *get_thread_pool_ptr() const; // Return nullptr in GPU implementation

    protected:
        static constexpr float infinity = std::numeric_limits<float>::infinity();

        const unsigned int n_inputs;
        const unsigned int n_neurons;
        const float tau_s;
        const float tau;
        const float threshold;

        SpikeArray post_spikes;
        EvSpikeSim::vector<unsigned int> n_spikes; // Counts number of post spikes per neuron

        EvSpikeSim::ndarray<float> weights;

        EvSpikeSim::vector<const Spike *> current_pre_spike; // Keeps track of pre spikes during inference
        EvSpikeSim::vector<float> a; // Sum w * exp(t/tau_s)
        EvSpikeSim::vector<float> b; // Sum w * exp(t/tau) - reset

        EvSpikeSim::vector<float> buffer; // Buffer for spike times
        unsigned int buffer_size;
        EvSpikeSim::unique_ptr<bool> buffer_full; // Set to true when a neuron's buffer is full

        KernelData kernel_data;
        infer_kernel_fct kernel_fct;
    };
}