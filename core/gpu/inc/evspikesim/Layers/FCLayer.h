//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <evspikesim/Layers/FCLayerDescriptor.h>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Misc/NDArray.h>

namespace EvSpikeSim {
    class FCLayer : public Layer {
    public:
        using descriptor_type = FCLayerDescriptor;

    public:
        FCLayer(const descriptor_type &desc, unsigned int buffer_size = 64u);

        FCLayer(const descriptor_type &desc, Initializer &initializer, unsigned int buffer_size = 64u);

        FCLayer(const descriptor_type &desc, Initializer &&initializer, unsigned int buffer_size = 64u);

        const SpikeArray &infer(const SpikeArray &pre_spikes) override;
    };
}