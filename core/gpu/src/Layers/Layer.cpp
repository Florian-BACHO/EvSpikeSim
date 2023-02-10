//
// Created by Florian Bacho on 23/01/23.
//

#include <evspikesim/Layers/Layer.h>

using namespace EvSpikeSim;

Layer::Layer(const LayerDescriptor &desc, const std::initializer_list<unsigned int> &weights_dims,
             unsigned int buffer_size) :
        desc(desc),
        post_spikes(),
        n_spikes(desc.n_neurons),
        weights(weights_dims),
        current_pre_spike(desc.n_neurons),
        a(desc.n_neurons),
        b(desc.n_neurons),
        buffer(desc.n_neurons * buffer_size),
        buffer_size(buffer_size),
        buffer_full(nullptr) {
    cudaMallocManaged((void **) &buffer_full, sizeof(bool));
}

Layer::Layer(const LayerDescriptor &desc, const std::initializer_list<unsigned int> &weights_dims,
             Initializer &initializer, unsigned int buffer_size) :
        desc(desc),
        post_spikes(),
        n_spikes(desc.n_neurons),
        weights(weights_dims, initializer),
        current_pre_spike(desc.n_neurons),
        a(desc.n_neurons),
        b(desc.n_neurons),
        buffer(desc.n_neurons * buffer_size),
        buffer_size(buffer_size),
        buffer_full(nullptr) {
    cudaMallocManaged((void **) &buffer_full, sizeof(bool));
}

Layer::~Layer() {
    cudaFree(buffer_full);
}

template<typename T>
__global__ void fill_kernel(T *ptr, std::size_t n, T fill_value) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        ptr[idx] = fill_value;
}

template<typename T>
static void fill(T *ptr, std::size_t n, T fill_value, int thread_per_block) {
    auto n_blocks = n / thread_per_block + (n % thread_per_block == 0 ? 0 : 1);

    fill_kernel << < n_blocks, thread_per_block >> > (ptr, n, fill_value);
}

static auto get_n_thread_per_block() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}

void Layer::reset(const SpikeArray &pre_spikes) {
    auto thread_per_block = get_n_thread_per_block();

    fill(current_pre_spike.data(), current_pre_spike.size(), pre_spikes.get_c_ptr(), thread_per_block);
    fill(a.data(), a.size(), 0.0f, thread_per_block);
    fill(b.data(), b.size(), 0.0f, thread_per_block);
    fill(n_spikes.data(), n_spikes.size(), 0u, thread_per_block);
    post_spikes.clear();
}

void Layer::reset_buffer() {
    auto thread_per_block = get_n_thread_per_block();

    fill(buffer.data(), buffer.size(), infinity, thread_per_block);
    *buffer_full = false;
}