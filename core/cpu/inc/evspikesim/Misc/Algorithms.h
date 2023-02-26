//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

namespace EvSpikeSim {
    /**
     * Fill an EvSpikeSim container with the given value.
     * @tparam ForwardIt The type of the iterators.
     * @tparam T The type of the value.
     * @param begin The begin iterator.
     * @param end The end iterator.
     * @param fill_value The value to fill.
     */
    template<class ForwardIt, class T>
    void fill(ForwardIt begin, ForwardIt end, const T &fill_value) {
        std::fill(begin, end, fill_value);
    }
}