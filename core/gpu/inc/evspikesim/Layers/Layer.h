//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <limits>
#include <initializer_list>
#include <evspikesim/SpikeArray.h>
#include <evspikesim/Misc/NDArray.h>
#include <evspikesim/Misc/CudaManagedAllocator.h>

namespace EvSpikeSim {
    class Layer {
    public:
        Layer(const std::initializer_list<unsigned int> &weights_dims,
              unsigned int n_inputs,
              unsigned int n_neurons,
              float tau_s,
              float threshold,
              Initializer &initializer,
              unsigned int buffer_size);

        virtual ~Layer();

        virtual const SpikeArray &infer(const SpikeArray &pre_spikes) = 0;

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

    protected:
        static constexpr float infinity = std::numeric_limits<float>::infinity();

        const unsigned int n_inputs;
        const unsigned int n_neurons;
        const float tau_s;
        const float tau;
        const float threshold;

        SpikeArray post_spikes;
        // Counts number of post spikes per neuron
        std::vector<unsigned int, CudaManagedAllocator < unsigned int>> n_spikes;

        NDArray<float, CudaManagedAllocator < float>> weights;

        // Keeps track of pre spikes during inference
        std::vector<const Spike *, CudaManagedAllocator<const Spike *>> current_pre_spike;
        std::vector<float, CudaManagedAllocator <float>> a; // Sum w * exp(t/tau_s)
        std::vector<float, CudaManagedAllocator <float>> b; // Sum w * exp(t/tau) - reset

        std::vector<float, CudaManagedAllocator <float>> buffer; // Buffer for spike times
        unsigned int buffer_size;
        bool *buffer_full; // Set to true when a neuron's buffer is full
    };
}