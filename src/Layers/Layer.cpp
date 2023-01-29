//
// Created by Florian Bacho on 23/01/23.
//

#include "Layers/Layer.h"

using namespace EvSpikeSim;

void Layer::reset() {
    std::fill(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 0.0f);
}

void Layer::reset_buffer() {
    std::fill(buffer.begin(), buffer.end(), infinity);
    buffer_full = false;
}