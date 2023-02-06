//
// Created by Florian Bacho on 25/01/23.
//

#pragma once

namespace EvSpikeSim {
    class Initializer {
    public:
        virtual ~Initializer() = default;

        virtual float operator()() = 0;
    };
}