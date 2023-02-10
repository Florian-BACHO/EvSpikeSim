//
// Created by Florian Bacho on 22/01/23.
//

#include <cstdio>
#include <cmath>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/Spike.h>

using namespace EvSpikeSim;

FCLayer::FCLayer(const descriptor_type &desc, unsigned int buffer_size) :
        Layer(desc, {desc.n_neurons, desc.n_inputs}, buffer_size) {}

FCLayer::FCLayer(const descriptor_type &desc, Initializer &initializer, unsigned int buffer_size) :
        Layer(desc, {desc.n_neurons, desc.n_inputs}, initializer, buffer_size) {}

FCLayer::FCLayer(const descriptor_type &desc, Initializer &&initializer,
                 unsigned int buffer_size) :
        FCLayer(desc, initializer, buffer_size) {}

struct KernelData {
    FCLayerDescriptor desc;
    const Spike **current_pre_spike;
    const Spike *end_pre_spikes;
    unsigned int *n_spikes;
    float *weights;
    float *a;
    float *b;
    float *buffer;
    bool *buffer_full;
    unsigned int buffer_size;
};

__device__ void apply_decay_and_weight(KernelData &kernel_data, unsigned int neuron_idx, float delta_t,
                                       unsigned int pre_idx) {
    float weight = kernel_data.weights[neuron_idx * kernel_data.desc.n_inputs + pre_idx];
    float exp_tau = exp(-delta_t / kernel_data.desc.tau);
    float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

    kernel_data.a[neuron_idx] = kernel_data.a[neuron_idx] * exp_tau_s + weight;
    kernel_data.b[neuron_idx] = kernel_data.b[neuron_idx] * exp_tau + weight;
}

__inline__ __device__ float compute_inside_x(float a, float b, float c) {
    return b * b - 4 * a * c;
}

__inline__ __device__ float compute_inside_log(float a, float b, float x) {
    return 2.0f * a / (b + x);
}

__inline__ __device__ float compute_spike_time(float inside_log, float tau) {
    return tau * log(inside_log);
}

__inline__ __device__ float apply_reset(float b, float c, float inside_log) {
    return b - c * inside_log;
}

__device__ bool fire(KernelData &kernel_data, unsigned int neuron_idx, float current_time,
                     float next_pre_time, unsigned int &n_spike_buffer) {
    bool valid_spike;
    float x, inside_log, spike_time;
    float a = kernel_data.a[neuron_idx];
    float *b = kernel_data.b + neuron_idx;
    float c = kernel_data.desc.c;
    float tau = kernel_data.desc.tau;
    float last_time = 0.0;

    while (n_spike_buffer < kernel_data.buffer_size) {
        // Compute spike time
        x = compute_inside_x(a, *b, c);
        if (x < 0)
            return false;
        x = sqrt(x);
        inside_log = compute_inside_log(a, *b, x);
        if (inside_log <= 0)
            return false;
        spike_time = compute_spike_time(inside_log, tau);

        // Check for valid spike
        valid_spike = last_time < spike_time && spike_time < next_pre_time;
        if (!valid_spike)
            return false;

        // Valid spike
        last_time = spike_time;
        kernel_data.n_spikes[neuron_idx]++;
        kernel_data.buffer[neuron_idx * kernel_data.buffer_size + n_spike_buffer] = current_time + spike_time;
        *b = apply_reset(*b, c, inside_log);
        n_spike_buffer++;

        // Reached the end of the buffer
        if (n_spike_buffer == kernel_data.buffer_size) {
            *(kernel_data.buffer_full) = true;
            return true;
        }
    }
    return false;
}

// Get the relative time to the next pre-synaptic spike
__inline__ __device__ float get_next_spike_time(const Spike *spike, const Spike *end) {
    return (spike == end - 1) ? (INFINITY) : ((spike + 1)->time - spike->time);
}

__global__ void infer_kernel(KernelData kernel_data, bool first_call) {
    auto neuron_idx = threadIdx.x;
    unsigned int n_spike_buffer = 0;
    float next_time = 0.0;
    const Spike **current_pre_spike = kernel_data.current_pre_spike + neuron_idx;

    if (!first_call && *current_pre_spike != kernel_data.end_pre_spikes) {
        next_time = get_next_spike_time(*current_pre_spike, kernel_data.end_pre_spikes);
        if (fire(kernel_data, neuron_idx, (*current_pre_spike)->time, next_time, n_spike_buffer))
            return;
        (*current_pre_spike)++;
    }
    while (*current_pre_spike != kernel_data.end_pre_spikes) {
        if (n_spike_buffer >= kernel_data.buffer_size)
            break;
        apply_decay_and_weight(kernel_data, neuron_idx, next_time, (*current_pre_spike)->index);
        next_time = get_next_spike_time(*current_pre_spike, kernel_data.end_pre_spikes);
        if (fire(kernel_data, neuron_idx, (*current_pre_spike)->time, next_time, n_spike_buffer))
            return;
        (*current_pre_spike)++;
    }
}

const SpikeArray &FCLayer::infer(const SpikeArray &pre_spikes) {
    KernelData kernel_data = {
            *static_cast<const FCLayerDescriptor *>(&desc),
            current_pre_spike.data(),
            pre_spikes.get_c_ptr() + pre_spikes.n_spikes(),
            n_spikes.data(),
            weights.get_c_ptr(),
            a.data(),
            b.data(),
            buffer.data(),
            buffer_full,
            buffer_size
    };
    bool first_call = true;

    reset(pre_spikes);

    if (pre_spikes.is_empty())
        return post_spikes;

    do {
        reset_buffer();
        cudaDeviceSynchronize();

        infer_kernel << < 1, desc.n_neurons >> > (kernel_data, first_call);
        cudaDeviceSynchronize();

        process_buffer();
        first_call = false;
    } while (*buffer_full);

    post_spikes.sort();
    return post_spikes;
}