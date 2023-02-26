//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <memory>
#include <vector>

namespace EvSpikeSim {
    template <typename T>
    using vector = std::vector<T>; /**< Type definition of the EvSpikeSim vector */

    template <typename T>
    using unique_ptr = std::unique_ptr<T>; /**< Type definition of the EvSpikeSim unique pointer */

    template <typename T>
    using shared_ptr = std::unique_ptr<T>; /**< Type definition of the EvSpikeSim shared pointer */

    template <typename T>
    auto make_unique = std::make_unique<T>; /**< Make unique function */

    template <typename T>
    auto make_shared = std::make_shared<T>; /**< Make shared function */
}