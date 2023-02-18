//
// Created by Florian Bacho on 16/02/23.
//

#pragma once

#include <cmath>
#include <evspikesim/Layers/InferKernelBase.h>
#include <evspikesim/Misc/ThreadPool.h>

namespace EvSpikeSim {
    void infer_range(KernelData &kernel_data, const Spike *end_pre_spikes,
                     unsigned int neuron_start, unsigned int neuron_end,
                     bool first_call) {
        for (unsigned int i = neuron_start; i < neuron_end && i < kernel_data.n_neurons; i++)
            infer_neuron(kernel_data, end_pre_spikes, i, first_call);
    }
}

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