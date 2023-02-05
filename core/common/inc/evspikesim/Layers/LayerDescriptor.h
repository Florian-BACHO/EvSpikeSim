//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

namespace EvSpikeSim {
    struct LayerDescriptor {
    public:
        LayerDescriptor(unsigned int n_inputs, unsigned int n_neurons, float tau_s, float threshold);

        bool operator==(const LayerDescriptor &rhs) const;

    public:
        const unsigned int n_inputs;
        const unsigned int n_neurons;
        const float tau_s;
        const float tau;
        const float threshold;
        const float c;
    };
}