//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <evspikesim/Misc/GPUDevice.h>

namespace EvSpikeSim {
    /**
     * GPU kernel used by fill.
     * @tparam T Type of the value to fill.
     * @param ptr The array to fill.
     * @param n The size of the array to fill.
     * @param fill_value The value to fill.
     */
    template<typename T>
    __global__ void fill_kernel(T *ptr, std::size_t n, T fill_value) {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n)
            ptr[idx] = fill_value;
    }

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
        T *begin_ptr = &(*begin);
        T *end_ptr = &(*end);
        std::size_t n = end_ptr - begin_ptr;
        static auto n_thread_per_block = get_n_thread_per_block();
        auto n_blocks = n / n_thread_per_block + (n % n_thread_per_block == 0 ? 0 : 1);

        fill_kernel<<<n_blocks, n_thread_per_block>>> (begin_ptr, n, fill_value);
    }
}