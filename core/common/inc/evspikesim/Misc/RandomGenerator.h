//
// Created by Florian Bacho on 05/02/23.
//

#pragma once

#include <random>
#include <ctime>

namespace EvSpikeSim {
    class RandomGenerator : public std::default_random_engine {
    public:
        RandomGenerator() { seed(get_time_seed()); }

        RandomGenerator(unsigned long s) { seed(s); }

    private:
        unsigned long get_time_seed() const { return static_cast<long unsigned int>(std::time(nullptr)); }
    };
}