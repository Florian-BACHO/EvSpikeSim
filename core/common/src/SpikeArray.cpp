//
// Created by Florian Bacho on 21/01/23.
//

#include <algorithm>
#include <evspikesim/SpikeArray.h>

using namespace EvSpikeSim;

SpikeArray::SpikeArray() : spikes(), sorted(true) {}

SpikeArray::SpikeArray(const std::vector<unsigned int> &indices, const std::vector<float> &times) :
        SpikeArray(indices.begin(), indices.end(), times.begin()) {}

void SpikeArray::add(unsigned int index, float time) {
    if (is_max_capacity())
        extend_capacity();
    spikes.emplace_back(index, time);
    sorted = false;
}

void SpikeArray::add(const std::vector<unsigned int> &indices, const std::vector<float> &times) {
    add(indices.begin(), indices.end(), times.begin());
}

void SpikeArray::sort() {
    if (is_sorted())
        return;
    std::sort(spikes.begin(), spikes.end());
    sorted = true;
}

void SpikeArray::clear() {
    spikes.clear();
}

SpikeArray::const_iterator SpikeArray::begin() const {
    return spikes.begin();
};

SpikeArray::const_iterator SpikeArray::end() const {
    return spikes.end();
};

std::size_t SpikeArray::n_spikes() const {
    return spikes.size();
}

bool SpikeArray::is_sorted() const {
    return sorted;
}

bool SpikeArray::empty() const {
    return n_spikes() == 0;
}

const Spike *SpikeArray::c_ptr() const {
    return spikes.data();
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

bool SpikeArray::is_max_capacity() const {
    return spikes.size() == spikes.capacity();
}

void SpikeArray::extend_capacity() {
    spikes.reserve(spikes.capacity() + block_size);
}