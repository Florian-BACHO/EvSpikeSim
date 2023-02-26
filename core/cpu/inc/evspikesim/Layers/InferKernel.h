//
// Created by Florian Bacho on 16/02/23.
//

#pragma once

#include <cmath>
#include <evspikesim/Layers/InferKernelDefinitions.h>
#include <evspikesim/Misc/ThreadPool.h>

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Disable documentation and avoids duplicate warning when building API documentation with exhale

namespace EvSpikeSim {
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