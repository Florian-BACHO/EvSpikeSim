//
// Created by Florian Bacho on 22/01/23.
//

#include <cstdio>
#include <cmath>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/Spike.h>

using namespace EvSpikeSim;

struct KernelData {
    const Spike *pre_spikes;
    std::size_t n_pre_spikes;
    FCLayerDescriptor desc;
    unsigned int *n_spikes;
    float *weights;
    float *a;
    float *b;
    float *buffer;
    unsigned int buffer_size;
};

void FCLayer::process_buffer() {
    float time;

    for (unsigned int i = 0; i < desc.n_neurons; i++) {
        for (unsigned int j = 0; j < buffer_size; j++) {
            time = buffer[i * buffer_size + j];
            if (time == infinity)
                break;
            post_spikes.add(i, time);
        }
    }
}

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

__device__ void fire(KernelData &kernel_data, unsigned int neuron_idx, float current_time,
                     float next_pre_time, unsigned int &n_spike_buffer) {
    bool valid_spike;
    float x, inside_log, spike_time;
    float a = kernel_data.a[neuron_idx];
    float *b = kernel_data.b + neuron_idx;
    float c = kernel_data.desc.c;
    float tau = kernel_data.desc.tau;
    float last_time = 0.0;

    while (n_spike_buffer < kernel_data.buffer_size) {
        x = compute_inside_x(a, *b, c);
        if (x < 0)
            return;
        x = sqrt(x);
        inside_log = compute_inside_log(a, *b, x);
        if (inside_log <= 0)
            return;
        spike_time = compute_spike_time(inside_log, tau);
        valid_spike = last_time < spike_time && spike_time < next_pre_time;
        if (!valid_spike)
            return;
        last_time = spike_time;
        kernel_data.n_spikes[neuron_idx]++;
        // Add current_time as spike time is relative to last pre-spike
        kernel_data.buffer[neuron_idx * kernel_data.buffer_size + n_spike_buffer] = current_time + spike_time;
        *b = apply_reset(*b, c, inside_log);
        n_spike_buffer++;
    }
}

__global__ void infer_kernel(KernelData kernel_data) {
    auto neuron_idx = threadIdx.x;
    unsigned int n_spike_buffer = 0;
    float next_time = 0.0;
    auto last_spike_idx = kernel_data.n_pre_spikes - 1;

    for (auto i = 0; i < kernel_data.n_pre_spikes; i++) {
        if (n_spike_buffer >= kernel_data.buffer_size)
            break;
        apply_decay_and_weight(kernel_data, neuron_idx, next_time, kernel_data.pre_spikes[i].index);
        next_time = (i == last_spike_idx) ? (INFINITY) : (kernel_data.pre_spikes[i + 1].time -
                kernel_data.pre_spikes[i].time);
        fire(kernel_data, neuron_idx, kernel_data.pre_spikes[i].time, next_time, n_spike_buffer);
    }
}

const SpikeArray &FCLayer::infer(const SpikeArray &pre_spikes) {
    KernelData kernel_data = {
            pre_spikes.c_ptr(),
            pre_spikes.n_spikes(),
            *static_cast<const FCLayerDescriptor *>(&desc),
            n_spikes.data(),
            weights.c_ptr(),
            a.data(),
            b.data(),
            buffer.data(),
            buffer_size
    };

    reset();
    cudaDeviceSynchronize();

    infer_kernel<<<1, desc.n_neurons>>>(kernel_data);
    cudaDeviceSynchronize();

    process_buffer();

    post_spikes.sort();
    return post_spikes;
}