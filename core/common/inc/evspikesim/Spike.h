//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <iostream>

namespace EvSpikeSim {
    /**
     * Spike event. Spikes are fired and received by spiking neurons. They drive the computation during inference.
     */
    struct Spike {
    public:
        /**
         * Default constructor.
         */
        Spike() = default;

        /**
         * Constructs a spike with a given index and timing.
         * @param index The index of the neuron that fired the spike.
         * @param time The timing of the spike.
         */
        Spike(unsigned int index, float time);

        // Comparators
        /**
         * Checks if two spikes occured within a time distance of Spike::epsilon,
         * i.e. std::abs(this->time - rhs.time) < epsilon.
         * @param rhs The other spike to compare.
         * @return True if the absolute difference between the two spike timings is bellow Spike::epsilon.
         */
        bool operator==(const Spike &rhs) const;

        /**
         * Checks if two spikes did not occur within a time distance of Spike::epsilon,
         * i.e. std::abs(this->time - rhs.time) >= epsilon.
         * @param rhs The other spike to compare.
         * @return True if the absolute difference between the two spike timings is above Spike::epsilon.
         */
        bool operator!=(const Spike &rhs) const;

        /**
         * Checks if the spike occured before a given spike, i.e. this->time < rhs.time.
         * @param rhs The other spike to compare.
         * @return True if the spike occured before rhs.
         */
        bool operator<(const Spike &rhs) const;

        /**
         * Checks if the spike occured before or at the same time as given spike, i.e. this->time <= rhs.time.
         * @param rhs The other spike to compare.
         * @return True if the spike occured before or at the same time as rhs.
         */
        bool operator<=(const Spike &rhs) const;

        /**
         * Checks if the spike occured after a given spike, i.e. this->time > rhs.time.
         * @param rhs The other spike to compare.
         * @return True if the spike occured after rhs.
         */
        bool operator>(const Spike &rhs) const;

        /**
         * Checks if the spike occured after or at the same time as given spike, i.e. this->time >= rhs.time.
         * @param rhs The other spike to compare.
         * @return True If the spike occured after or at the same time as rhs.
         */
        bool operator>=(const Spike &rhs) const;


        /**
         * Formats the spike to the given output stream.
         * @param os The output stream to write.
         * @param spike The spike to format.
         * @return the Output stream os.
         */
        // IOStream
        friend std::ostream &operator<<(std::ostream &os, const Spike &spike);

    public:
        static constexpr float epsilon = 1e-6; /**< Time difference within two spikes are considered equal. */
        unsigned int index; /**< Index (in the layer) of the neuron that fired the spike. */
        float time; /**< Timing of the spike. */
    };


    std::ostream &operator<<(std::ostream &os, const Spike &spike);
}