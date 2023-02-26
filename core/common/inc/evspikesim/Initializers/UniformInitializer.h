//
// Created by Florian Bacho on 25/01/23.
//

#pragma once

#include <random>
#include <evspikesim/Initializers/Initializer.h>

namespace EvSpikeSim {
    /**
     * Initializer that initializes with a uniform distribution.
     * @tparam Generator Type of the random engine. It is compatible with random engine in standard library.
     */
    template<class Generator>
    class UniformInitializer : public Initializer {
    public:
        /**
         * Constructs the initializer with the given random generator, lower bound and upper bound.
         * @param generator Random engine.
         * @param lower_bound Lower bound of the uniform distribution.
         * @param upper_bound Upper bound of the uniform distribution.
         */
        explicit UniformInitializer(Generator &generator, float lower_bound = -1.0f, float upper_bound = 1.0f) :
                generator(generator), distribution(lower_bound, upper_bound) {}

        /**
         * Call operator that generates values for weights initialization.
         * @return A new value following a uniform distribution.
         */
        inline float operator()() override { return distribution(generator); }

    private:
        Generator &generator; /**< Random engine used for random generation */
        std::uniform_real_distribution<float> distribution; /**< Uniform distribution used to generate numbers */
    };
}