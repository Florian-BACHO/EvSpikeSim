//
// Created by Florian Bacho on 23/01/23.
//

#include "Layers/FCLayer.h"
#include "SpikingNetwork.h"

using namespace EvSpikeSim;

const SpikeArray& SpikingNetwork::infer(const SpikeArray &pre_spikes) {
    auto *spikes = &pre_spikes;

    for (auto &layer : layers)
        spikes = &layer->infer(*spikes);

    return *spikes;
}