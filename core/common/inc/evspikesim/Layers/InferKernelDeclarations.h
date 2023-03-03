//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <evspikesim/Misc/Decorators.h>
#include <evspikesim/Misc/ContainerTypes.h>
#include <evspikesim/Spike.h>

namespace EvSpikeSim {
#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Disable documentation with exhale
    /**
     * Internal structured used to pass data to the inference kernel in a way that is both compatible with CPU threads
     * and GPU kernels.
     */
    struct KernelData {
        unsigned int n_inputs;
        unsigned int n_neurons;
        float tau_s;
        float tau;
        float threshold;

        unsigned int *n_spikes;

        float *weights;

        const Spike **current_pre_spike;
        float *current_time;
        float *a;
        float *b;

        float *buffer;
        bool *buffer_full;
        unsigned int buffer_size;

        float *synaptic_traces_tau;
        float *neuron_traces_tau;
        float *synaptic_traces;
        float *neuron_traces;
        unsigned int n_synaptic_traces;
        unsigned int n_neuron_traces;
    };

#endif

    /**
     * Callback used to update the neuron traces at each pre-synaptic event.
     * @param weight The weight of the connection that received the pre-synaptic spike.
     * @param neuron_traces The neuron traces.
     */
    CALLBACK void on_pre_neuron(float weight, float *neuron_traces);

    /**
     * Callback used to update the synaptic traces at each pre-synaptic event.
     * @param weight The weight of the connection that received the pre-synaptic spike.
     * @param neuron_traces The neuron traces.
     * @param synaptic_traces The traces of the synapse that received the pre-synaptic spike.
     */
    CALLBACK void on_pre_synapse(float weight, const float *neuron_traces, float *synaptic_traces);

    /**
     * Callback used to update the neuron traces at each post-synaptic event.
     * @param neuron_traces The neuron traces.
     */
    CALLBACK void on_post_neuron(float *neuron_traces);

    /**
     * Callback used to the update synaptic traces at each post-synaptic event.
     * @param weight The weight of the synapse that is being updated.
     * @param neuron_traces The neuron traces.
     * @param synaptic_traces The traces of the synapse that is being updated.
     */
    CALLBACK void on_post_synapse(float weight, const float *neuron_traces, float *synaptic_traces);
}

// Suppress return-type-c-linkage warning as we know that only C++ code will use these functions
#ifndef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"
#endif

// Exposed extern C functions

/**
 * Gets the time constants of each synaptic trace and each neuron trace respectively.
 *
 * @param tau_s The synaptic time constant of the neurons.
 * @param tau The membrane time constant of the neurons (2 * tau_s).
 * @return A pair of vectors containing the time constants of each synaptic trace and each neuron trace respectively.
 */
extern "C" std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> get_traces_tau(float tau_s, float tau);

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
                             bool first_call, void *thread_pool_ptr);

#ifndef __CUDACC__
#pragma GCC diagnostic pop
#endif

namespace EvSpikeSim {
    /**
     * The signature of the extern C get_traces_tau function.
     */
    using get_traces_tau_fct = decltype(&get_traces_tau);
    /**
     * The signature of the extern C infer_kernel function.
     */
    using infer_kernel_fct = decltype(&infer_kernel);

    // Symbol definitions
    /**
     * Symbol of the extern C get_traces_tau function.
     */
    static constexpr char get_traces_tau_symbol[] = "get_traces_tau";
    /**
     * Symbol of the extern C infer_kernel function.
     */
    static constexpr char infer_kernel_symbol[] = "infer_kernel";
}