//
// Created by Florian Bacho on 16/02/23.
//

#pragma once

#include <cmath>
#include <evspikesim/SpikeArray.h>
#include <evspikesim/Misc/NDArray.h>
#include <evspikesim/Misc/ThreadPool.h>

#ifndef CALLBACK
#define CALLBACK
#endif

namespace EvSpikeSim {
    struct FCKernelData {
        unsigned int n_inputs;
        unsigned int n_neurons;
        float tau_s;
        float tau;
        float threshold;

        std::vector<SpikeArray::const_iterator> &current_pre_spike;
        SpikeArray::const_iterator end_pre_spikes;
        std::vector<unsigned int> &n_spikes;

        NDArray<float> &weights;

        std::vector<float> &a;
        std::vector<float> &b;

        std::vector<float> &buffer;
        bool *buffer_full;
        unsigned int buffer_size;
    };

// Provide kernel sources if KERNEL_SOURCE_DEFINITION is defined
#ifdef KERNEL_SOURCE_DEFINITION
    // Callback declarations
    CALLBACK float on_pre(const Spike &pre_spike, float weight);

    static inline float compute_inside_x(float a, float b, float c) {
        return b * b - 4 * a * c;
    }

    static inline float compute_inside_log(float a, float b, float x) {
        return 2.0f * a / (b + x);
    }

    static inline float compute_spike_time(float inside_log, float tau) {
        return tau * std::log(inside_log);
    }

    static inline float apply_reset(float b, float c, float inside_log) {
        return b - c * inside_log;
    }

    void apply_decay_and_weight(FCKernelData &kernel_data, unsigned int neuron_idx, float delta_t,
                                const Spike &spike) {
        float weight = on_pre(spike, kernel_data.weights.get(neuron_idx, spike.index));
        float exp_tau = exp(-delta_t / kernel_data.tau);
        float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

        kernel_data.a[neuron_idx] = kernel_data.a[neuron_idx] * exp_tau_s + weight;
        kernel_data.b[neuron_idx] = kernel_data.b[neuron_idx] * exp_tau + weight;
    }

    static inline float get_next_time(const SpikeArray::const_iterator &current,
                                      const SpikeArray::const_iterator &end) {
        return ((current + 1) == end) ? (INFINITY) : ((current + 1)->time - current->time);
    }

    bool fire(FCKernelData &kernel_data, unsigned int neuron_idx, float current_time, float next_pre_time,
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

    void infer_neuron(FCKernelData &kernel_data, unsigned int neuron_idx, bool first_call) {
        float next_time = 0.0;
        unsigned int n_spike_buffer = 0;
        SpikeArray::const_iterator &spike = kernel_data.current_pre_spike[neuron_idx];

        // Carry on inference if buffer was full in previous call
        if (!first_call && spike != kernel_data.end_pre_spikes) {
            next_time = get_next_time(spike, kernel_data.end_pre_spikes);
            if (fire(kernel_data, neuron_idx, spike->time, next_time, n_spike_buffer))
                return;
            spike++;
        }
        while (spike != kernel_data.end_pre_spikes) {
            apply_decay_and_weight(kernel_data, neuron_idx, next_time, *spike);
            next_time = get_next_time(spike, kernel_data.end_pre_spikes);
            if (fire(kernel_data, neuron_idx, spike->time, next_time, n_spike_buffer))
                return;
            spike++;
        }
    }

    void infer_range(FCKernelData &kernel_data, unsigned int neuron_start, unsigned int neuron_end,
                     bool first_call) {
        for (unsigned int i = neuron_start; i < neuron_end && i < kernel_data.n_neurons; i++)
            infer_neuron(kernel_data, i, first_call);
    }

#endif
}

extern "C" void fc_layer_kernel(std::shared_ptr<EvSpikeSim::ThreadPool> thread_pool,
                                EvSpikeSim::FCKernelData &kernel_data,
                                bool first_call);

// Provide kernel sources if KERNEL_SOURCE_DEFINITION is defined
#ifdef KERNEL_SOURCE_DEFINITION
extern "C" void fc_layer_kernel(std::shared_ptr<EvSpikeSim::ThreadPool> thread_pool,
                                EvSpikeSim::FCKernelData &kernel_data,
                                bool first_call) {
    unsigned int n_neurons_per_thread = std::max(kernel_data.n_neurons / thread_pool->get_thread_count(), 1u);
    std::vector<std::future<void>> tasks;

    for (auto i = 0u; i < kernel_data.n_neurons; i += n_neurons_per_thread)
        tasks.push_back(thread_pool->submit([&kernel_data, i, n_neurons_per_thread, first_call] {
            infer_range(kernel_data, i, i + n_neurons_per_thread, first_call);
        }));

    // Wait for end of task
    for (auto &it : tasks)
        it.get();
}
#endif