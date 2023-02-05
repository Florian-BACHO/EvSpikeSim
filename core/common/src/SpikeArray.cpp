//
// Created by Florian Bacho on 21/01/23.
//

#include <algorithm>
#include <evspikesim/SpikeArray.h>

using namespace EvSpikeSim;

SpikeArray::SpikeArray(const std::vector<unsigned int> &indices, const std::vector<float> &times) :
        SpikeArray(indices.begin(), indices.end(), times.begin()) {}

void SpikeArray::add(unsigned int index, float time) {
    if (is_max_capacity())
        extend_capacity();
    spikes.emplace_back(index, time);
}

void SpikeArray::add(const std::vector<unsigned int> &indices, const std::vector<float> &times) {
    add(indices.begin(), indices.end(), times.begin());
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

std::ostream &EvSpikeSim::operator<<(std::ostream &os, const SpikeArray &array) {
    for (const auto &spike : array)
        os << spike << std::endl;
    return os;
}