//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <memory>
#include <vector>
#include <evspikesim/Misc/NDArray.h>

namespace EvSpikeSim {
    template <typename T>
    using vector = std::vector<T>;

    template <typename T>
    using ndarray = NDArray<T>;

    template <typename T>
    using unique_ptr = std::unique_ptr<T>;

    template <typename T>
    using shared_ptr = std::unique_ptr<T>;

    template <typename T>
    auto make_unique = std::make_unique<T>;

    template <typename T>
    auto make_shared = std::make_shared<T>;
}