//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <evspikesim/Layers/LayerDescriptor.h>

namespace EvSpikeSim {
    struct FCLayerDescriptor : public LayerDescriptor {
    public:
        FCLayerDescriptor(unsigned int n_inputs, unsigned int n_neurons, float tau_s, float threshold) :
                LayerDescriptor(n_inputs, n_neurons, tau_s, threshold) {}
    };
}