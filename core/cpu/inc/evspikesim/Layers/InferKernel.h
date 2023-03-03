//
// Created by Florian Bacho on 16/02/23.
//

#pragma once

#include <iostream>
#include <cmath>
#include <evspikesim/Layers/InferKernelDefinitions.h>
#include <evspikesim/Misc/ThreadPool.h>

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Disable documentation and avoids duplicate warning when building API documentation with exhale

namespace EvSpikeSim {
    /**
     * Updates the time of neuron and synaptic traces, i.e. applies decay.
     * @param kernel_data The inference data.
     * @param delta_t The time elapsed since the last event.
     */
    DEVICE void update_traces_time(KernelData &kernel_data, float delta_t) {
        float exp_tau;

        for (auto i = 0u; i < kernel_data.n_neuron_traces; i++) {
            exp_tau = compute_exp_tau(delta_t, kernel_data.neuron_traces_tau[i]);
            kernel_data.neuron_traces[i] *= exp_tau;
        }

        // Update synaptic traces
        for (auto i = 0u; i < kernel_data.n_synaptic_traces; i++) {
            exp_tau = compute_exp_tau(delta_t, kernel_data.synaptic_traces_tau[i]);
            for (auto j = 0u; j < kernel_data.n_inputs; j++)
                kernel_data.synaptic_traces[j * kernel_data.n_synaptic_traces + i] *= exp_tau;
        }
    }

    /**
     * Calls the on-pre callbacks with a given pre-synaptic spike.
     * @param kernel_data The inference data.
     * @param spike The pre-synaptic event that occured.
     */
    DEVICE void call_on_pre(KernelData &kernel_data, const Spike *spike) {
        float weight = kernel_data.weights[spike->index];

        on_pre_synapse(weight, kernel_data.neuron_traces,
                       kernel_data.synaptic_traces + spike->index * kernel_data.n_synaptic_traces);
        on_pre_neuron(weight, kernel_data.neuron_traces);
    }

    /**
     * Calls the on-post callbacks.
     * @param kernel_data The inference data
     */
    DEVICE void call_on_post(KernelData &kernel_data) {
        on_post_neuron(kernel_data.neuron_traces);

        if (kernel_data.n_synaptic_traces == 0)
            return;
        for (auto i = 0u; i < kernel_data.n_inputs; i++)
            on_post_synapse(kernel_data.weights[i], kernel_data.neuron_traces,
                            kernel_data.synaptic_traces + i * kernel_data.n_synaptic_traces);
    }

    /**
     * Infer a range of neurons.
     * @param kernel_data The inference data.
     * @param end_pre_spikes The end pointer of the pre-synaptic spikes.
     * @param neuron_start The index of the first neuron to infer.
     * @param neuron_end The index of the end neuron.
     * @param first_call Must be true if this is the first call to the kernel during the inference, otherwise false.
     */
    void infer_range(KernelData &kernel_data, const Spike *end_pre_spikes,
                     unsigned int neuron_start, unsigned int neuron_end,
                     bool first_call) {
        for (unsigned int i = neuron_start; i < neuron_end && i < kernel_data.n_neurons; i++)
            infer_neuron(kernel_data, end_pre_spikes, i, first_call);
    }
}

/**
 * Inference kernel function.
 *
 * @param kernel_data The data required by the inference kernel.
 * @param end_pre_spikes The end of the pre-synaptic spikes.
 * @param first_call Must be true if this is the first call to the kernel during the inference, otherwise false.
 * @param thread_pool_ptr The pointer to the global thread pool if the implementation is for CPU. For GPU, nullptr is
 * passed and the argument is unused.
 */
extern "C" void infer_kernel(EvSpikeSim::KernelData &kernel_data, const EvSpikeSim::Spike *end_pre_spikes,
                             bool first_call, void *thread_pool_ptr) {
    EvSpikeSim::ThreadPool *thread_pool = reinterpret_cast<EvSpikeSim::ThreadPool *>(thread_pool_ptr);
    unsigned int n_neurons_per_thread = std::max(kernel_data.n_neurons / thread_pool->get_thread_count(), 1u);
    std::vector<std::future<void>> tasks;

    for (auto i = 0u; i < kernel_data.n_neurons; i += n_neurons_per_thread)
        tasks.push_back(thread_pool->submit([&kernel_data, end_pre_spikes, i, n_neurons_per_thread, first_call] {
            infer_range(kernel_data, end_pre_spikes, i, i + n_neurons_per_thread, first_call);
        }));

    // Wait for end of task
    for (auto &it : tasks)
        it.get();
}

#endif