//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Layers/InferKernelDeclarations.h>
#include <evspikesim/Misc/NDArray.h>

namespace EvSpikeSim {
    class FCLayer : public Layer {
    public:
        FCLayer(unsigned int n_inputs,
                unsigned int n_neurons,
                float tau_s,
                float threshold,
                Initializer &initializer,
                unsigned int buffer_size = 64u,
                infer_kernel_fct kernel = infer_kernel);
    };
}