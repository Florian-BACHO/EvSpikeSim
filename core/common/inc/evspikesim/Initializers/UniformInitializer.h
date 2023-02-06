//
// Created by Florian Bacho on 25/01/23.
//

#pragma once

#include <random>
#include <evspikesim/Initializers/Initializer.h>

namespace EvSpikeSim {
    template<class Generator>
    class UniformInitializer : public Initializer {
    public:
        explicit UniformInitializer(Generator &generator, float lower_bound = -1.0f, float upper_bound = 1.0f) :
                generator(generator), distribution(lower_bound, upper_bound) {}

        inline float operator()() override { return distribution(generator); }

    private:
        Generator &generator;
        std::uniform_real_distribution<float> distribution;
    };
}