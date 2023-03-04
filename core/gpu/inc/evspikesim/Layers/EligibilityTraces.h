//
// Created by Florian Bacho on 16/02/23.
//

#pragma once

#include <evspikesim/Layers/InferKernelDefinitions.h>

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Disable documentation and avoids duplicate warning when building API documentation with exhale

namespace EvSpikeSim {
    /**
     * The cuda kernel that applies decay to the the neuron and synaptic traces.
     * @param kernel_data The inference data.
     * @param delta_t The time elapsed since the last event.
     */
    GLOBAL void update_traces_time_kernel(KernelData kernel_data, float delta_t) {
        auto synapse_idx = threadIdx.x;
        auto trace_idx = blockIdx.x;
        float exp_tau = compute_exp_tau(delta_t, kernel_data.synaptic_traces_tau[trace_idx]);

        kernel_data.synaptic_traces[synapse_idx * kernel_data.n_synaptic_traces + trace_idx] *= exp_tau;

        if (synapse_idx > 0 || trace_idx > 0)
            return;

        // Update neuron traces
        for (auto i = 0u; i < kernel_data.n_neuron_traces; i++) {
            exp_tau = compute_exp_tau(delta_t, kernel_data.neuron_traces_tau[i]);
            kernel_data.neuron_traces[i] *= exp_tau;
        }
    }

    /**
     * Updates the time of neuron and synaptic traces, i.e. applies decay. It uses dynamic parallelism to update traces
     * in parallel of the inference. Stream semantic ensures that child kernels are updated in order of call.
     * @param kernel_data The inference data.
     * @param delta_t The time elapsed since the last event.
     */
    DEVICE void update_traces_time(KernelData &kernel_data, float delta_t) {
        EvSpikeSim::update_traces_time_kernel << < kernel_data.n_synaptic_traces, kernel_data.n_inputs >> >
                                                                                  (kernel_data, delta_t);
    }

    /**
     * The CUDA kernel that calls the on-pre callbacks.
     * @param weight The weight of the connection that received the pre-synaptic spike.
     * @param neuron_traces The neuron traces.
     * @param synaptic_traces The traces of the synapse that received the pre-synaptic spike.
     */
    GLOBAL void call_on_pre_kernel(float weight, float *neuron_traces, float *synaptic_traces) {
        on_pre_synapse(weight, neuron_traces, synaptic_traces);
        __syncthreads();
        if (threadIdx.x == 0)
            on_pre_neuron(weight, neuron_traces);
    }

    /**
     * Calls the on-pre callbacks with a given pre-synaptic spike. It uses dynamic parallelism to update traces in
     * parallel of the inference. Stream semantic ensures that child kernels are updated in order of call.
     * @param kernel_data The inference data.
     * @param spike The pre-synaptic event that occured.
     */
    INLINE DEVICE void call_on_pre(KernelData &kernel_data, const Spike *spike) {
        float weight = kernel_data.weights[spike->index];

        EvSpikeSim::call_on_pre_kernel << < 1, 1 >> > (weight, kernel_data.neuron_traces,
                kernel_data.synaptic_traces + spike->index * kernel_data.n_synaptic_traces);
    }

    /**
     * The CUDA kernel that calls the on-post callbacks.
     * @param weights The weights of the neuron.
     * @param neuron_traces The neuron traces.
     * @param synaptic_traces The synaptic traces.
     * @param n_synaptic_traces The number of trace per synapse.
     */
    GLOBAL void call_on_post_kernel(const float *weights, float *neuron_traces, float *synaptic_traces,
                                    unsigned int n_synaptic_traces) {
        unsigned int synapse_idx = threadIdx.x;

        if (synapse_idx == 0)
            on_post_neuron(neuron_traces);
        __syncthreads();
        on_post_synapse(weights[synapse_idx], neuron_traces, synaptic_traces + synapse_idx * n_synaptic_traces);
    }

    /**
     * Calls the on-post callbacks. It uses dynamic parallelism to update traces in parallel of the inference.
     * Stream semantic ensures that child kernels are updated in order of call.
     * @param kernel_data The inference data
     */
    INLINE DEVICE void call_on_post(KernelData &kernel_data) {
        EvSpikeSim::call_on_post_kernel << < 1, kernel_data.n_inputs >> > (kernel_data.weights,
                kernel_data.neuron_traces, kernel_data.synaptic_traces, kernel_data.n_synaptic_traces);
    }

    /**
     * The CUDA inference kernel.
     * @param kernel_data The inference data.
     * @param end_pre_spikes The end of the pre-synaptic spikes.
     * @param first_call Must be true if this is the first call to the kernel during the inference, otherwise false.
     */
    GLOBAL void kernel_gpu(KernelData kernel_data, const Spike *end_pre_spikes, bool first_call) {
        auto neuron_idx = threadIdx.x;

        infer_neuron(kernel_data, end_pre_spikes, neuron_idx, first_call);
    }
}

/**
 * Inference kernel function.
 *
 * @param kernel_data The data required by the inference kernel.
 * @param end_pre_spikes The end of the pre-synaptic spikes.
 * @param first_call Must be true if this is the first call to the kernel during the inference, otherwise false.
 * @param unused The pointer to the global thread pool if the implementation is for CPU. For GPU, nullptr is
 * passed and the argument is unused.
 */
extern "C" void infer_kernel(EvSpikeSim::KernelData &kernel_data, const EvSpikeSim::Spike *end_pre_spikes,
                             bool first_call, void *unused) {
    (void) unused;
    cudaDeviceSynchronize();
    EvSpikeSim::kernel_gpu << < 1, kernel_data.n_neurons >> > (kernel_data, end_pre_spikes, first_call);
    cudaDeviceSynchronize();
}

#endif