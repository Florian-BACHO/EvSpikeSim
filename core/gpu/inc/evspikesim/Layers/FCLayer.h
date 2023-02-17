//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Layers/FCLayerKernel.h>
#include <evspikesim/Misc/NDArray.h>

namespace EvSpikeSim {
    class FCLayer : public Layer {
    public:
        // Signature and symbol of extern "C" kernel in user-defined shared library
        using kernel_signature = void(*)(FCKernelData &, bool);
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
        kernel_signature kernel;
    };
}