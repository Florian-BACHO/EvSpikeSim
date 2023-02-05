//
// Created by Florian Bacho on 23/01/23.
//

#include <evspikesim/Layers/Layer.h>

using namespace EvSpikeSim;

template <typename T>
__global__ void fill_kernel(T* ptr, std::size_t n, T fill_value) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        ptr[idx] = fill_value;
}

template <typename T>
static void fill(T* ptr, std::size_t n, T fill_value, int thread_per_block) {
    int n_blocks = n / thread_per_block + (n % thread_per_block == 0 ? 0 : 1);

    fill_kernel<<<n_blocks, thread_per_block>>>(ptr, n, fill_value);
}

void Layer::reset() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int thread_per_block = prop.maxThreadsPerBlock;

    fill(a.data(), a.size(), 0.0f, thread_per_block);
    fill(b.data(), b.size(), 0.0f, thread_per_block);
    fill(buffer.data(), buffer.size(), infinity, thread_per_block);
    fill(n_spikes.data(), n_spikes.size(), 0u, thread_per_block);
    post_spikes.clear();
}