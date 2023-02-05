//
// Created by Florian Bacho on 25/01/23.
//

#pragma once

#include <random>
#include <evspikesim/Initializers/Initializer.h>

namespace EvSpikeSim {
    template <class Generator>
    class NormalInitializer : public Initializer {
    public:
        NormalInitializer(Generator &generator, float mean = 0.0f, float stddev = 1.0f) :
                generator(generator), distribution(mean, stddev) {}

        inline float operator()() override { return distribution(generator); }

    private:
        Generator &generator;
        std::normal_distribution<float> distribution;
    };
}