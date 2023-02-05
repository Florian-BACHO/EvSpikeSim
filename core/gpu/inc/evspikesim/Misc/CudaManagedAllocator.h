//
// Created by Florian Bacho on 29/01/23.
//

#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace EvSpikeSim {
    template<typename T>
    class CudaManagedAllocator {
    public:
        using size_type = size_t;
        using pointer = T *;
        using const_pointer = const T *;
        using value_type = T;

        template<typename _Tp1>
        struct rebind {
            typedef CudaManagedAllocator<_Tp1> other;
        };

    public:
        CudaManagedAllocator() = default;
        CudaManagedAllocator(const CudaManagedAllocator &other) = default;
        ~CudaManagedAllocator() = default;

        pointer allocate(size_type n, const void *hint = 0) {
            (void)hint; // Unused
            pointer ptr;

            cudaMallocManaged((void **)&ptr, n * sizeof(T));
            return ptr;
        }

        void deallocate(pointer p, size_type n) {
            (void)n; // Unused

            cudaFree(p);
        }
    };
}