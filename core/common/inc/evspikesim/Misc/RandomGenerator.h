//
// Created by Florian Bacho on 05/02/23.
//

#pragma once

#include <random>
#include <ctime>

namespace EvSpikeSim {
    /**
     * This class extends the std::default_random_engine to be able to set seeds more easily.
     */
    class RandomGenerator : public std::default_random_engine {
    public:
        /**
         * Constructs a random generator with the seed set as the current timestamp.
         */
        RandomGenerator() { seed(get_time_seed()); }

        /**
         * Constructs a random generator with a given seed.
         * @param s The seed of the random engine.
         */
        RandomGenerator(unsigned long s) { seed(s); }

    private:
        /**
         * Creates a seed based on the current timestamp.
         * @return The current time
         */
        unsigned long get_time_seed() const { return static_cast<long unsigned int>(std::time(nullptr)); }
    };
}