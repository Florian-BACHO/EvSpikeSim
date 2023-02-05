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
        template <typename... Args>
        FCLayer(const FCLayerDescriptor &desc, Args... args) :
                Layer(desc, {desc.n_neurons, desc.n_inputs}, args...) {}

        const SpikeArray &infer(const SpikeArray &pre_spikes) override;

    private:

        void process_buffer();
    };
}