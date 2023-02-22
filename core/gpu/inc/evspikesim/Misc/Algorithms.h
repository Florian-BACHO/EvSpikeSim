//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <evspikesim/Misc/GPUDevice.h>

namespace EvSpikeSim {
    template<typename T>
    __global__ void fill_kernel(T *ptr, std::size_t n, T fill_value) {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n)
            ptr[idx] = fill_value;
    }

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