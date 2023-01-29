//
// Created by Florian Bacho on 22/01/23.
//

#include "Spike.h"

using namespace EvSpikeSim;

Spike::Spike(unsigned int index, float time) : index(index), time(time) {}

bool Spike::operator==(const Spike &rhs) const {
    return std::abs(time - rhs.time) < epsilon;
}

bool Spike::operator!=(const Spike &rhs) const {
    return std::abs(time - rhs.time) > epsilon;
}

bool Spike::operator<(const Spike &rhs) const {
    return time < rhs.time;
}

bool Spike::operator<=(const Spike &rhs) const {
    return time <= rhs.time;
}

bool Spike::operator>(const Spike &rhs) const {
    return time > rhs.time;
}

bool Spike::operator>=(const Spike &rhs) const {
    return time >= rhs.time;
}

std::ostream& EvSpikeSim::operator<<(std::ostream &os, const Spike &spike) {
    return os << "Index: " << spike.index << ", Time: " << spike.time;
}