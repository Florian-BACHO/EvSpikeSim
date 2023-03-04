//
// Created by Florian Bacho on 25/01/23.
//

#pragma once

namespace EvSpikeSim {
    /**
     * Interface for initializers. Defined as a functor.
     */
    class Initializer {
    public:
        /**
         * Default destructor.
         */
        virtual ~Initializer() = default;

        /**
         * Call operator that generates values for weights initialization.
         * @return A new value.
         */
        virtual float operator()() = 0;
    };
}