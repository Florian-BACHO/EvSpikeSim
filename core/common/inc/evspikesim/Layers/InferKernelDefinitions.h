//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <evspikesim/Layers/InferKernelDeclarations.h>

namespace EvSpikeSim {
    INLINE DEVICE float compute_exp_tau(float delta, float tau) {
        return exp(delta / tau);
    }

    INLINE DEVICE float get_next_time(const Spike *current, const Spike *end) {
        return ((current + 1) == end) ? (INFINITY) : ((current + 1)->time);
    }

    INLINE DEVICE float compute_inside_x(float a, float b, float c) {
        return b * b - 4 * a * c;
    }

    INLINE DEVICE float compute_inside_log(float a, float b, float x) {
        return 2.0f * a / (b + x);
    }

    INLINE DEVICE float compute_spike_time(float inside_log, float tau) {
        return tau * std::log(inside_log);
    }

    DEVICE void update_time(KernelData &kernel_data, unsigned int neuron_idx, float new_time) {
        float delta_t = kernel_data.current_time[neuron_idx] - new_time; // Delta t is negative for decay
        float exp_tau = compute_exp_tau(delta_t, kernel_data.tau);
        float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s
        float *traces;

        // Update membrane potential
        kernel_data.a[neuron_idx] *= exp_tau_s;
        kernel_data.b[neuron_idx] *= exp_tau;

        // Update neuron traces
        traces = kernel_data.neuron_traces + neuron_idx * kernel_data.n_neuron_traces;
        for (auto i = 0u; i < kernel_data.n_neuron_traces; i++) {
            exp_tau = compute_exp_tau(delta_t, kernel_data.neuron_traces_tau[i]);
            traces[i] *= exp_tau;
        }

        // Update synaptic traces
        traces = kernel_data.synaptic_traces + neuron_idx * kernel_data.n_inputs * kernel_data.n_synaptic_traces;
        for (auto i = 0u; i < kernel_data.n_synaptic_traces; i++) {
            exp_tau = compute_exp_tau(delta_t, kernel_data.synaptic_traces_tau[i]);
            for (auto j = 0u; j < kernel_data.n_inputs; j++)
                traces[j * kernel_data.n_synaptic_traces + i] *= exp_tau;
        }

        // Update time
        kernel_data.current_time[neuron_idx] = new_time;
    }

    DEVICE void integrate_pre_spike(KernelData &kernel_data, unsigned int neuron_idx, const Spike &spike) {
        // On-pre callback
        float weight = on_pre(spike, kernel_data.weights[neuron_idx * kernel_data.n_inputs + spike.index],
                              kernel_data.neuron_traces + kernel_data.n_neuron_traces * neuron_idx,
                              kernel_data.synaptic_traces + (neuron_idx * kernel_data.n_inputs + spike.index) *
                              kernel_data.n_synaptic_traces,
                              kernel_data.n_inputs);

        // Integrate weights
        kernel_data.a[neuron_idx] += weight;
        kernel_data.b[neuron_idx] += weight;
    }

    DEVICE bool fire(KernelData &kernel_data, unsigned int neuron_idx, const Spike *current_pre_spike,
                     const Spike *end_pre_spikes, unsigned int &n_spike_buffer) {
        float next_pre_time = get_next_time(current_pre_spike, end_pre_spikes);
        bool valid_spike;
        float x, inside_log, spike_time;
        float &a = kernel_data.a[neuron_idx];
        float &b = kernel_data.b[neuron_idx];
        float &current_time = kernel_data.current_time[neuron_idx];

        while (n_spike_buffer < kernel_data.buffer_size) {
            // Compute spike time
            x = compute_inside_x(a, b, kernel_data.threshold);
            if (x < 0)
                return false;
            x = sqrt(x);
            inside_log = compute_inside_log(a, b, x);
            if (inside_log <= 0)
                return false;
            spike_time = current_time + compute_spike_time(inside_log, kernel_data.tau);

            // Check for valid spike
            valid_spike = current_time < spike_time && spike_time < next_pre_time;
            if (!valid_spike)
                return false;

            // Valid spike
            update_time(kernel_data, neuron_idx, spike_time);
            on_post(kernel_data.neuron_traces + kernel_data.n_neuron_traces * neuron_idx,
                    kernel_data.synaptic_traces + kernel_data.n_synaptic_traces * kernel_data.n_inputs * neuron_idx,
                    kernel_data.n_inputs);
            kernel_data.n_spikes[neuron_idx]++;
            kernel_data.buffer[neuron_idx * kernel_data.buffer_size + n_spike_buffer] = spike_time;
            b -= kernel_data.threshold;
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
        unsigned int n_spike_buffer = 0; // Keeps track of how many post-synaptic spikes times are in buffer
        const Spike *&current_pre_spike = kernel_data.current_pre_spike[neuron_idx];

        // Carry on inference if buffer was full in previous call
        if (!first_call && current_pre_spike != end_pre_spikes) {
            if (fire(kernel_data, neuron_idx, current_pre_spike, end_pre_spikes, n_spike_buffer))
                return; // Buffer full
            current_pre_spike++;
        }
        while (current_pre_spike != end_pre_spikes) {
            update_time(kernel_data, neuron_idx, current_pre_spike->time);
            integrate_pre_spike(kernel_data, neuron_idx, *current_pre_spike);
            if (fire(kernel_data, neuron_idx, current_pre_spike, end_pre_spikes, n_spike_buffer))
                return; // Buffer full
            current_pre_spike++;
        }
    }
}