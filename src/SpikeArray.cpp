//
// Created by Florian Bacho on 21/01/23.
//

#include "SpikeArray.h"

using namespace EvSpikeSim;

void SpikeArray::add(unsigned int index, float time) {
    if (is_max_capacity())
        extend_capacity();
    spikes.emplace_back(index, time);
}

void SpikeArray::sort() {
    std::sort(spikes.begin(), spikes.end());
}

void SpikeArray::clear() {
    spikes.clear();
}

bool SpikeArray::operator==(const SpikeArray &rhs) const {
    return spikes == rhs.spikes;
}

bool SpikeArray::operator!=(const SpikeArray &rhs) const {
    return spikes != rhs.spikes;
}

std::ostream& EvSpikeSim::operator<<(std::ostream &os, const SpikeArray &array) {
    for (const auto &spike : array)
        os << spike << std::endl;
    return os;
}