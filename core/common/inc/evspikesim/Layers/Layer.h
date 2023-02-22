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
              get_traces_tau_fct traces_tau_fct = get_traces_tau,
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

        inline auto get_n_synaptic_traces() const { return static_cast<unsigned int>(synaptic_traces_tau.size()); }

        inline auto get_n_neuron_traces() const { return static_cast<unsigned int>(neuron_traces_tau.size()); }

        inline const auto &get_synaptic_traces() const { return synaptic_traces; };

        inline const auto &get_neuron_traces() const { return neuron_traces; };

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

        // Outputs
        SpikeArray post_spikes;
        EvSpikeSim::vector<unsigned int> n_spikes; // Counts number of post spikes per neuron

        // Connectionns
        EvSpikeSim::ndarray<float> weights;

        // Inference states
        EvSpikeSim::vector<const Spike *> current_pre_spike; // Keeps track of pre spikes during inference
        EvSpikeSim::vector<float> current_time; // Keeps track of simulation time of each neuron
        EvSpikeSim::vector<float> a; // Sum w * exp(t/tau_s)
        EvSpikeSim::vector<float> b; // Sum w * exp(t/tau) - reset

        // Buffer
        EvSpikeSim::vector<float> buffer; // Buffer for spike times
        unsigned int buffer_size;
        EvSpikeSim::unique_ptr<bool> buffer_full; // Set to true when a neuron's buffer is full

        // Traces
        EvSpikeSim::vector<float> synaptic_traces_tau;
        EvSpikeSim::vector<float> neuron_traces_tau;
        EvSpikeSim::vector<float> synaptic_traces;
        EvSpikeSim::vector<float> neuron_traces;

        // Inference kernel
        KernelData kernel_data;
        infer_kernel_fct kernel_fct;

    private:
        void init_traces(get_traces_tau_fct traces_tau_fct);

        KernelData get_kernel_data();

        void *get_thread_pool_ptr() const; // Return nullptr in GPU implementation
    };
}