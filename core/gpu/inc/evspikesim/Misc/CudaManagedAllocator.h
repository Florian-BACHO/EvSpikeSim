//
// Created by Florian Bacho on 29/01/23.
//

#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace EvSpikeSim {
    /**
     * Wrapper around cudaMallocManaged to create cuda-managed vectors.
     * @tparam T Type allocated by the allocator.
     */
    template<typename T>
    class CudaManagedAllocator {
    public:
        using size_type = std::size_t; /**< Type of a size */
        using pointer = T *; /**< Type of the allocated pointer */
        using const_pointer = const T *; /**< Type of the allocated const pointer */
        using value_type = T; /**< Type of the value */

        /**
         * Rebind of CudaManagedAllocator.
         * @tparam _Tp1 Type allocated by the other allocator.
         */
        template<typename _Tp1>
        struct rebind {
            typedef CudaManagedAllocator<_Tp1> other; /**< Type of other allocator */
        };

    public:
        /**
         * Default constructor.
         */
        CudaManagedAllocator() = default;

        /**
         * Copy constructor (default).
         * @param other The other allocator to copy.
         */
        CudaManagedAllocator(const CudaManagedAllocator &other) = default;

        /**
         * Default destructor.
         */
        ~CudaManagedAllocator() = default;

        /**
         * Allocates n values of type T.
         * @param n The size to allocate.
         * @param hint Unused.
         * @return The allocated pointer.
         */
        pointer allocate(size_type n, const void *hint = 0) const {
            (void)hint; // Unused
            pointer ptr;

            cudaMallocManaged((void **)&ptr, n * sizeof(T));
            return ptr;
        }

        /**
         * Free the given pointer.
         * @param p The pointer to free.
         * @param n Unused.
         */
        void deallocate(pointer p, size_type n) const {
            (void)n; // Unused

            cudaFree(p);
        }

        /**
         * Compares allocators.
         * @param other The other allocator to compare.
         * @return True
         */
        bool operator==(const CudaManagedAllocator<T> &other) const {
            (void) other;

            return true;
        }

        /**
         * Compares allocators.
         * @param other The other allocator to compare.
         * @return False
         */
        bool operator!=(const CudaManagedAllocator<T> &other) const {
            (void) other;

            return false;
        }
    };
}