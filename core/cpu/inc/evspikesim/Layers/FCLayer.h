//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Layers/FCLayerKernel.h>
#include <evspikesim/Misc/NDArray.h>
#include <evspikesim/Misc/ThreadPool.h>

namespace EvSpikeSim {
    class FCLayer : public Layer {
    public:
        // Signature and symbol of extern "C" kernel in user-defined shared library
        using kernel_signature = void(*)(std::shared_ptr<ThreadPool>, FCKernelData &, bool);
        static constexpr char kernel_symbol[] = "fc_layer_kernel";

    public:
        FCLayer(unsigned int n_inputs,
                unsigned int n_neurons,
                float tau_s,
                float threshold,
                Initializer &initializer,
                unsigned int buffer_size = 64u,
                kernel_signature kernel = fc_layer_kernel);

        const SpikeArray &infer(const SpikeArray &pre_spikes) override;

    private:
        void apply_decay_and_weight(unsigned int neuron_idx, float delta_t, const Spike &spike);

        bool fire(unsigned int neuron_idx, float current_time, float next_pre_time, unsigned int &n_spike_buffer);

        void infer_neuron(const SpikeArray &pre_spikes, unsigned int neuron_start, bool first_call);

        void infer_range(const SpikeArray &pre_spikes, unsigned int neuron_start, unsigned int neuron_end,
                         bool first_call);

    private:
        kernel_signature kernel;
    };
}