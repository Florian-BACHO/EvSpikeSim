//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

namespace EvSpikeSim {
    template<class ForwardIt, class T>
    void fill(ForwardIt begin, ForwardIt end, const T &fill_value) {
        std::fill(begin, end, fill_value);
    }
}