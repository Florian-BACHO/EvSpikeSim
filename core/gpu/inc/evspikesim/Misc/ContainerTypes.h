//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <evspikesim/Misc/CudaManagedAllocator.h>

namespace EvSpikeSim {
    template <typename T>
    using vector = std::vector<T, CudaManagedAllocator<T>>;  /**< Type definition of the EvSpikeSim vector for GPU */

    template <typename T>
    using unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;  /**< Type definition of the EvSpikeSim unique pointer for GPU */

    template <typename T>
    using shared_ptr = std::unique_ptr<T, std::function<void(T *)>>;  /**< Type definition of the EvSpikeSim shared pointer for GPU */

    /**
     * Instanciates a unique_ptr of the given type with the given arguments.
     * @tparam T Type to instanciate.
     * @tparam Args Types of the constructor arguments.
     * @param args The arguments to forward to the constructor.
     * @return A unique_ptr containing the newly created object of type T.
     */
    template <typename T, typename... Args>
    unique_ptr<T> make_unique(Args... args) {
        CudaManagedAllocator<T> allocator;
        T *ptr = allocator.allocate(1);
        auto deleter = [&allocator](T *ptr) {
            allocator.deallocate(ptr, std::size_t(1));
        };

        return {ptr, deleter};
    }

    /**
     * Instanciates a shared_ptr of the given type with the given arguments.
     * @tparam T Type to instanciate.
     * @tparam Args Types of the constructor arguments.
     * @param args The arguments to forward to the constructor.
     * @return A shared_ptr containing the newly created object of type T.
     */
    template <typename T, typename... Args>
    shared_ptr<T> make_shared(Args... args) {
        CudaManagedAllocator<T> allocator;
        T *ptr = allocator.allocate(1);
        auto deleter = [allocator](T *ptr) {
            allocator.deallocate(ptr, 1);
        };

        return {ptr, deleter};
    }
}