//
// Created by Florian Bacho on 25/01/23.
//

#pragma once

#include <evspikesim/Initializers/Initializer.h>

namespace EvSpikeSim {
    class ConstantInitializer : public Initializer {
    public:
        ConstantInitializer(float value = 0.0f) : value(value) {}

        inline float operator()() override { return value; }

    private:
        const float value;
    };
}