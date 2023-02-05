//
// Created by Florian Bacho on 23/01/23.
//

#include <evspikesim/Layers/Layer.h>

using namespace EvSpikeSim;

void Layer::reset() {
    std::fill(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 0.0f);
    std::fill(buffer.begin(), buffer.end(), infinity);
    post_spikes.clear();
    std::fill(n_spikes.begin(), n_spikes.end(), 0);
}