//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <evspikesim/Misc/Decorators.h>
#include <evspikesim/Misc/ContainerTypes.h>
#include <evspikesim/Spike.h>

namespace EvSpikeSim {
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

    // Signature definitons
    using get_traces_tau_fct = std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> (*)(float, float);
    using infer_kernel_fct = void (*)(KernelData &, const Spike *, bool, void *);

    // Symbol definitions
    static constexpr char get_traces_tau_symbol[] = "get_traces_tau";
    static constexpr char infer_kernel_symbol[] = "infer_kernel";


    // Callback declarations
    CALLBACK float on_pre(const Spike &pre_spike, float weight, float *neuron_traces, float *synaptic_traces,
                          unsigned int n_synapses);

    CALLBACK void on_post(float *neuron_traces, float *synaptic_traces, unsigned int n_synapses);
}

// Suppress return-type-c-linkage warning as we know that only C++ code will use these functions
#ifndef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"
#endif

// Exposed extern C functions

// Get time constants of each synaptic trace and each neuron trace respectively
extern "C" std::pair<EvSpikeSim::vector<float>, EvSpikeSim::vector<float>> get_traces_tau(float tau_s, float tau);

extern "C" void infer_kernel(EvSpikeSim::KernelData &kernel_data, const EvSpikeSim::Spike *end_pre_spikes,
                             bool first_call, void *thread_pool_ptr);

#ifndef __CUDACC__
#pragma GCC diagnostic pop
#endif