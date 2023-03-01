//
// Created by Florian Bacho on 16/02/23.
//

#pragma once

#include <evspikesim/Layers/InferKernelDefinitions.h>

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Disable documentation and avoids duplicate warning when building API documentation with exhale

namespace EvSpikeSim {
    GLOBAL void kernel_gpu(KernelData kernel_data, const Spike *end_pre_spikes, bool first_call) {
        auto neuron_idx = threadIdx.x;

        infer_neuron(kernel_data, end_pre_spikes, neuron_idx, first_call);
    }
}

extern "C" void infer_kernel(EvSpikeSim::KernelData &kernel_data, const EvSpikeSim::Spike *end_pre_spikes,
                             bool first_call, void *unused) {
    (void)unused;
    cudaDeviceSynchronize();
    EvSpikeSim::kernel_gpu<<<1, kernel_data.n_neurons>>>(kernel_data, end_pre_spikes, first_call);
    cudaDeviceSynchronize();
}

#endif