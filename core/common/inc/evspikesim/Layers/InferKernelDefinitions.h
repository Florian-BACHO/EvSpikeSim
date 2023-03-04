//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <evspikesim/Layers/InferKernelDeclarations.h>

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Disable documentation and avoids duplicate warning when building API documentation with exhale

namespace EvSpikeSim {
    /**
     * Computes an exponential decay.
     * @param delta The time elapsed.
     * @param tau The time constant of the decay.
     * @return The corresponding decay between 0 and 1.
     */
    INLINE DEVICE float compute_exp_tau(float delta, float tau) {
        return exp(delta / tau);
    }

    /**
     * Updates the time of neuron and synaptic traces, i.e. applies decay.
     * @param kernel_data The inference data.
     * @param delta_t The time elapsed since the last event.
     */
    DEVICE void update_traces_time(KernelData &kernel_data, float delta_t);

    /**
     * Calls the on-pre callbacks with a given pre-synaptic spike.
     * @param kernel_data The inference data.
     * @param spike The pre-synaptic event that occured.
     */
    DEVICE void call_on_pre(KernelData &kernel_data, const Spike *spike);

    /**
     * Calls the on-post callbacks.
     * @param kernel_data The inference data
     */
    DEVICE void call_on_post(KernelData &kernel_data);

    /**
     * Updates the time of the simulation. Applies exponential decays on the membrane potential and the eligibility
     * traces.
     *
     * @param kernel_data The inference data.
     * @param neuron_idx The index of the post-synaptic neuron being simulated.
     * @param new_time The new simulation time.
     */
    DEVICE void update_time(KernelData &kernel_data, float new_time) {
        float delta_t = *kernel_data.current_time - new_time; // Delta t is negative for decay
        float exp_tau = compute_exp_tau(delta_t, kernel_data.tau);
        float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

        // Update membrane potential
        *kernel_data.a *= exp_tau_s;
        *kernel_data.b *= exp_tau;

        update_traces_time(kernel_data, delta_t);

        // Update time
        *kernel_data.current_time = new_time;
    }

    /**
     * Gets the timing of the next pre-synaptic spike. If the last pre-synaptic is spiked, INFINITY is returned.
     * @param current A pointer on the current spike.
     * @param end The end of the spikes.
     * @return The timing of the next spike if current is not the last spike. Otherwise returns INFINITY.
     */
    INLINE DEVICE float get_next_time(const Spike *current, const Spike *end) {
        return ((current + 1) == end) ? (INFINITY) : ((current + 1)->time);
    }

    /**
     * Computes the square root of the quadratic determinant, i.e. x = sqrt(b^2 - 4ac).
     * @param a The factor a of the quadratic equation.
     * @param b The factor b of the quadratic equation.
     * @param c The factor c of the quadratic equation.
     * @return The square root of the quadratic determinant.
     */
    INLINE DEVICE float compute_inside_x(float a, float b, float c) {
        return b * b - 4 * a * c;
    }

    /**
     * Computes the term inside the logarithm operation of the spike time equation, i.e. 2 * a / (b + x).
     * @param a The factor a of the quadratic equation.
     * @param b The factor b of the quadratic equation.
     * @param x The square root of the quadratic determinant, computed by compute_inside_x.
     * @return The term inside the logarithm operation of the spike time equation.
     */
    INLINE DEVICE float compute_inside_log(float a, float b, float x) {
        return 2.0f * a / (b + x);
    }

    /**
     * Computes the spike time.
     * @param inside_log The term inside the logarithm operation of the spike time equation. This is computed by
     * compute_inside_log.
     * @param tau The membrane time constant.
     * @return The post-synaptic spike timing.
     */
    INLINE DEVICE float compute_spike_time(float inside_log, float tau) {
        return tau * log(inside_log);
    }

    /**
     * Integrates a post-synaptic spike into the membrane potential. Calls the on_pre callback with the
     * given pre-synaptic spike.
     *
     * @param kernel_data The inference data.
     * @param neuron_idx The index of the post-synaptic neuron being simulated.
     * @param spike The pre-synaptic spike to be integrated.
     */
    DEVICE void integrate_pre_spike(KernelData &kernel_data, const Spike &spike) {
        // On-pre callback
        float weight = kernel_data.weights[spike.index];

        // Integrate weights
        *kernel_data.a += weight;
        *kernel_data.b += weight;
    }

    /**
     * Fire the post-synaptic neuron after integrating the pre-synaptic spike (spike integration must be performed
     * prior to the call to fire). The function makes the neuron fire until the next pre-synaptic spike or until the
     * post-synaptic spike times buffer is full. In the later case, the buffer_full variable in the kernel_data is set
     * to true.
     * @param kernel_data The inference data.
     * @param neuron_idx The index of the post-synaptic neuron being simulated.
     * @param current_pre_spike The current pre-synaptic spike being processed.
     * @param end_pre_spikes The end of pre-synaptic spikes.
     * @param n_spike_buffer The number of post-synaptic spike times contained in the neuron's buffer.
     * @return True if the spike buffer of the neuron is full. False otherwise.
     */
    DEVICE bool fire(KernelData &kernel_data, const Spike *end_pre_spikes, unsigned int &n_spike_buffer) {
        float next_pre_time = get_next_time(*kernel_data.current_pre_spike, end_pre_spikes);
        bool valid_spike;
        float x, inside_log, spike_time;
        float &a = *kernel_data.a;
        float &b = *kernel_data.b;
        float &current_time = *kernel_data.current_time;

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
            update_time(kernel_data, spike_time);

            call_on_post(kernel_data);

            (*kernel_data.n_spikes)++;
            kernel_data.buffer[n_spike_buffer] = spike_time;
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

    DEVICE void offset_kernel_data(KernelData &kernel_data, unsigned int neuron_idx) {
        kernel_data.n_spikes += neuron_idx;
        kernel_data.weights += neuron_idx * kernel_data.n_inputs;
        kernel_data.current_pre_spike += neuron_idx;
        kernel_data.current_time += neuron_idx;
        kernel_data.a += neuron_idx;
        kernel_data.b += neuron_idx;
        kernel_data.buffer += neuron_idx * kernel_data.buffer_size;
        kernel_data.neuron_traces += neuron_idx * kernel_data.n_neuron_traces;
        kernel_data.synaptic_traces += neuron_idx * kernel_data.n_inputs * kernel_data.n_synaptic_traces;
    }

    /**
     * Infer the neuron at the given index.
     * @param kernel_data The inference data.
     * @param end_pre_spikes The end of pre-synaptic spikes.
     * @param neuron_idx The index of the post-synaptic neuron being simulated.
     * @param first_call Must be true if this is the first call to the kernel during the inference, otherwise false.
     */
    DEVICE void infer_neuron(KernelData kernel_data, const Spike *end_pre_spikes, unsigned int neuron_idx,
                             bool first_call) {
        unsigned int n_spike_buffer = 0; // Keeps track of how many post-synaptic spikes times are in buffer

        offset_kernel_data(kernel_data, neuron_idx);
        // Carry on inference if buffer was full in previous call
        if (!first_call && *kernel_data.current_pre_spike != end_pre_spikes) {
            if (fire(kernel_data, end_pre_spikes, n_spike_buffer))
                return; // Buffer full
            (*kernel_data.current_pre_spike)++;
        }
        while (*kernel_data.current_pre_spike != end_pre_spikes) {
            update_time(kernel_data, (*kernel_data.current_pre_spike)->time);
            integrate_pre_spike(kernel_data, **kernel_data.current_pre_spike);
            call_on_pre(kernel_data, *kernel_data.current_pre_spike);
            if (fire(kernel_data, end_pre_spikes, n_spike_buffer))
                return; // Buffer full
            (*kernel_data.current_pre_spike)++;
        }
    }


    /**
     * Gets the time constants of each synaptic trace and each neuron trace respectively.
     *
     * @param tau_s The synaptic time constant of the neurons.
     * @param tau The membrane time constant of the neurons (2 * tau_s).
     * @return A pair of vectors containing the time constants of each synaptic trace and each neuron trace respectively.
     */
    extern "C" std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> get_traces_tau(float tau_s, float tau) {
        return {synaptic_traces_tau(tau_s, tau), neuron_traces_tau(tau)};
    }
}

#endif