//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <vector>
#include <evspikesim/Misc/NDArray.h>
#include <evspikesim/Misc/CudaManagedAllocator.h>

namespace EvSpikeSim {
    template <typename T>
    using vector = std::vector<T, CudaManagedAllocator<T>>;

    template <typename T>
    using ndarray = NDArray<T, CudaManagedAllocator<T>>;

    template <typename T>
    using unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;

    template <typename T>
    using shared_ptr = std::unique_ptr<T, std::function<void(T *)>>;

    template <typename T, typename... Args>
    unique_ptr<T> make_unique(Args... args) {
        CudaManagedAllocator<T> allocator;
        T *ptr = allocator.allocate(1);
        auto deleter = [&allocator](T *ptr) {
            allocator.deallocate(ptr, std::size_t(1));
        };

        return {ptr, deleter};
    }

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