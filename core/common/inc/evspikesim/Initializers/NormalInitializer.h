//
// Created by Florian Bacho on 25/01/23.
//

#pragma once

#include <random>
#include <evspikesim/Initializers/Initializer.h>

namespace EvSpikeSim {
    /**
     * Initializer that initializes with a normal distribution.
     * @tparam Generator Type of the random engine. It is compatible with random engine in standard library.
     */
    template<class Generator>
    class NormalInitializer : public Initializer {
    public:
        /**
         * Constructs the initializer with the given random generator, mean and standard deviation.
         * @param generator Random engine.
         * @param mean Mean of the normal distribution.
         * @param stddev Standard deviation of the normal distribution.
         */
        explicit NormalInitializer(Generator &generator, float mean = 0.0f, float stddev = 1.0f) :
                generator(generator), distribution(mean, stddev) {}

        /**
         * Call operator that generates values for weights initialization.
         * @return A new value following a normal distribution.
         */
        inline float operator()() override { return distribution(generator); }

    private:
        Generator &generator; /**< Random engine used for random generation */
        std::normal_distribution<float> distribution; /**< Normal distribution used to generate numbers */
    };
}