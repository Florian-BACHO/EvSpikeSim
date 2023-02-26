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
     * The signature of the extern C get_traces_tau function.
     */
    using get_traces_tau_fct = std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> (*)(float, float);
    /**
     * The signature of the extern C infer_kernel function.
     */
    using infer_kernel_fct = void (*)(KernelData &, const Spike *, bool, void *);

    // Symbol definitions
    /**
     * Symbol of the extern C get_traces_tau function.
     */
    static constexpr char get_traces_tau_symbol[] = "get_traces_tau";
    /**
     * Symbol of the extern C infer_kernel function.
     */
    static constexpr char infer_kernel_symbol[] = "infer_kernel";


    // Callback declarations
    /**
     * Callback called at each pre-synaptic event during the inference of a neuron.
     *
     * @param pre_spike The pre-synaptic spike that caused the event.
     * @param weight The weight of the connection between the post-synaptic neuron and the pre-synaptic neuron that
     * caused the event.
     * @param neuron_traces The traces of the neuron that received the spike. The size of the array corresponds to the
     * number of neuron traces time constants returned by get_traces_tau.
     * @param synaptic_traces The traces of the synapse that received the spike. The size of the array corresponds to
     * the number of synaptic traces time constants returned by get_traces_tau.
     * @param n_synapses The number of synapses of the neuron.
     * @return The weight of the synapse to be integrated into the membrane potential.
     */
    CALLBACK float on_pre(const Spike &pre_spike, float weight, float *neuron_traces, float *synaptic_traces,
                          unsigned int n_synapses);

    /**
     * Callback called at each post-synaptic event during the inference of a neuron.
     *
     * @param neuron_traces The traces of the neuron that fired the spike. The size of the array corresponds to the
     * number of neuron traces time constants returned by get_traces_tau.
     * @param synaptic_traces The traces of all synapse of the neuron that fired the spike. The size of the array
     * corresponds to n_synapses times the number of synaptic traces time constants returned by get_traces_tau.
     * @param n_synapses The number of synapses of the neuron that fired the spike.
     */
    CALLBACK void on_post(float *neuron_traces, float *synaptic_traces, unsigned int n_synapses);
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