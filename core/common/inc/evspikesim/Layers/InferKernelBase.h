//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <evspikesim/Layers/InferKernelDeclarations.h>

namespace EvSpikeSim {
    INLINE DEVICE

    float compute_inside_x(float a, float b, float c) {
        return b * b - 4 * a * c;
    }

    INLINE DEVICE

    float compute_inside_log(float a, float b, float x) {
        return 2.0f * a / (b + x);
    }

    INLINE DEVICE

    float compute_spike_time(float inside_log, float tau) {
        return tau * std::log(inside_log);
    }

    INLINE DEVICE

    float apply_reset(float b, float c, float inside_log) {
        return b - c * inside_log;
    }

    INLINE DEVICE

    float get_next_time(const Spike *current, const Spike *end) {
        return ((current + 1) == end) ? (INFINITY) : ((current + 1)->time - current->time);
    }

    DEVICE void apply_decay_and_weight(KernelData &kernel_data, unsigned int neuron_idx, float delta_t,
                                       const Spike &spike) {
        float weight = on_pre(spike, kernel_data.weights[neuron_idx * kernel_data.n_inputs + spike.index]);
        float exp_tau = exp(-delta_t / kernel_data.tau);
        float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

        kernel_data.a[neuron_idx] = kernel_data.a[neuron_idx] * exp_tau_s + weight;
        kernel_data.b[neuron_idx] = kernel_data.b[neuron_idx] * exp_tau + weight;
    }

    DEVICE bool fire(KernelData &kernel_data, unsigned int neuron_idx, float current_time, float next_pre_time,
                     unsigned int &n_spike_buffer) {
        bool valid_spike;
        float x, inside_log, spike_time;
        float a = kernel_data.a[neuron_idx];
        float &b = kernel_data.b[neuron_idx];
        float last_time = 0.0;

        while (n_spike_buffer < kernel_data.buffer_size) {
            // Compute spike time
            x = compute_inside_x(a, b, kernel_data.threshold);
            if (x < 0)
                return false;
            x = sqrt(x);
            inside_log = compute_inside_log(a, b, x);
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
            b = apply_reset(b, kernel_data.threshold, inside_log);
            n_spike_buffer++;

            // Reached the end of the buffer
            if (n_spike_buffer == kernel_data.buffer_size) {
                *(kernel_data.buffer_full) = true;
                return true;
            }
        }
        return false;
    }

    DEVICE void infer_neuron(KernelData &kernel_data, const Spike *end_pre_spikes, unsigned int neuron_idx,
                             bool first_call) {
        float next_time = 0.0;
        unsigned int n_spike_buffer = 0;
        const Spike *&current_pre_spike = kernel_data.current_pre_spike[neuron_idx];

        // Carry on inference if buffer was full in previous call
        if (!first_call && current_pre_spike != end_pre_spikes) {
            next_time = get_next_time(current_pre_spike, end_pre_spikes);
            if (fire(kernel_data, neuron_idx, current_pre_spike->time, next_time, n_spike_buffer))
                return;
            current_pre_spike++;
        }
        while (current_pre_spike != end_pre_spikes) {
            apply_decay_and_weight(kernel_data, neuron_idx, next_time, *current_pre_spike);
            next_time = get_next_time(current_pre_spike, end_pre_spikes);
            if (fire(kernel_data, neuron_idx, current_pre_spike->time, next_time, n_spike_buffer))
                return;
            current_pre_spike++;
        }
    }
}