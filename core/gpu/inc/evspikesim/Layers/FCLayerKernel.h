//
// Created by Florian Bacho on 16/02/23.
//

#pragma once

#include <evspikesim/Spike.h>

#ifndef CALLBACK
#define CALLBACK __device__
#endif

namespace EvSpikeSim {
    struct FCKernelData {
        unsigned int n_inputs;
        unsigned int n_neurons;
        float tau_s;
        float tau;
        float threshold;

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

// Provide kernel sources if KERNEL_SOURCE_DEFINITION is defined
#ifdef KERNEL_SOURCE_DEFINITION
    // Callback declarations
    CALLBACK float on_pre(const Spike &pre_spike, float weight);

    __device__ void apply_decay_and_weight(FCKernelData &kernel_data, unsigned int neuron_idx, float delta_t,
                                           const Spike *spike) {
        float weight = on_pre(*spike, kernel_data.weights[neuron_idx * kernel_data.n_inputs + spike->index]);
        float exp_tau = exp(-delta_t / kernel_data.tau);
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

    __device__ bool fire(FCKernelData &kernel_data, unsigned int neuron_idx, float current_time,
                         float next_pre_time, unsigned int &n_spike_buffer) {
        bool valid_spike;
        float x, inside_log, spike_time;
        float a = kernel_data.a[neuron_idx];
        float *b = kernel_data.b + neuron_idx;
        float last_time = 0.0;

        while (n_spike_buffer < kernel_data.buffer_size) {
            // Compute spike time
            x = compute_inside_x(a, *b, kernel_data.threshold);
            if (x < 0)
                return false;
            x = sqrt(x);
            inside_log = compute_inside_log(a, *b, x);
            if (inside_log <= 0)
                return false;
            spike_time = compute_spike_time(inside_log, kernel_data.tau);

            // Check for valid spike
            valid_spike = last_time < spike_time && spike_time < next_pre_time;
            if (!valid_spike)
                return false;

            // Valid spike
            last_time = spike_time;
            kernel_data.n_spikes[neuron_idx]++;
            kernel_data.buffer[neuron_idx * kernel_data.buffer_size + n_spike_buffer] = current_time + spike_time;
            *b = apply_reset(*b, kernel_data.threshold, inside_log);
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

    __global__ void fc_layer_kernel_gpu(FCKernelData kernel_data, bool first_call) {
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
            apply_decay_and_weight(kernel_data, neuron_idx, next_time, *current_pre_spike);
            next_time = get_next_spike_time(*current_pre_spike, kernel_data.end_pre_spikes);
            if (fire(kernel_data, neuron_idx, (*current_pre_spike)->time, next_time, n_spike_buffer))
                return;
            (*current_pre_spike)++;
        }
    }
#endif
}

extern "C" void fc_layer_kernel(EvSpikeSim::FCKernelData &kernel_data, bool first_call);

// Provide kernel sources if KERNEL_SOURCE_DEFINITION is defined
#ifdef KERNEL_SOURCE_DEFINITION
extern "C" void fc_layer_kernel(EvSpikeSim::FCKernelData &kernel_data, bool first_call) {
    EvSpikeSim::fc_layer_kernel_gpu << < 1, kernel_data.n_neurons >> > (kernel_data, first_call);
    cudaDeviceSynchronize();
}
#endif