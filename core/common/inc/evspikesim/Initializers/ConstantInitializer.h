//
// Created by Florian Bacho on 25/01/23.
//

#pragma once

#include <evspikesim/Initializers/Initializer.h>

namespace EvSpikeSim {
    /**
     * Initializer that initializes with constant values.
     */
    class ConstantInitializer : public Initializer {
    public:
        /**
         * Constructs the initializer with the given constant value.
         * @param value The constant used for initialization.
         */
        explicit ConstantInitializer(float value = 0.0f) : value(value) {}

        /**
         * Call operator that generates values for weights initialization.
         * @return The constant given at construction.
         */
        inline float operator()() override { return value; }

    private:
        const float value; /**< Constant value for initialization */
    };
}