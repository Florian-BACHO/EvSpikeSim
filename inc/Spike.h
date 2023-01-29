//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <iostream>

namespace EvSpikeSim {
    struct Spike {
    public:
        Spike(unsigned int index, float time);

        // Comparators
        bool operator==(const Spike &rhs) const;

        bool operator!=(const Spike &rhs) const;

        bool operator<(const Spike &rhs) const;

        bool operator<=(const Spike &rhs) const;

        bool operator>(const Spike &rhs) const;

        bool operator>=(const Spike &rhs) const;

    public:
        static constexpr float epsilon = 1e-6;
        unsigned int index;
        float time;
    };

    // IOStream
    std::ostream& operator<<(std::ostream& os, const Spike &spike);
}