//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Layers/InferKernelDeclarations.h>
#include <evspikesim/Misc/NDArray.h>

namespace EvSpikeSim {
    /**
     * A layer of Fully-Connected (FC) spiking neurons.
     */
    class FCLayer : public Layer {
    public:
        /**
         * Constructs a fully-connected layer with the given parameters.
         * @param n_inputs The number of input neurons.
         * @param n_neurons The number of neurons in the layer.
         * @param tau_s The synaptic time constant.
         * @param threshold The threshold.
         * @param initializer The initializer to use to initialize weights.
         * @param buffer_size The size (per neuron) of the post-synaptic spike times buffer used during inference.
         * @param traces_tau_fct The function returning the time constants of synaptic and neuron eligibility traces.
         * @param kernel_fct The inference kernel function.
         */
        FCLayer(unsigned int n_inputs,
                unsigned int n_neurons,
                float tau_s,
                float threshold,
                Initializer &initializer,
                unsigned int buffer_size = 64u,
                get_traces_tau_fct traces_tau_fct = get_traces_tau,
                infer_kernel_fct kernel_fct = infer_kernel);
    };
}