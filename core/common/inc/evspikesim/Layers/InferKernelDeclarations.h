//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

#include <evspikesim/Misc/Decorators.h>
#include <evspikesim/Spike.h>

namespace EvSpikeSim {
    struct KernelData {
        unsigned int n_inputs;
        unsigned int n_neurons;
        float tau_s;
        float tau;
        float threshold;

        const Spike **current_pre_spike;
        unsigned int *n_spikes;

        float *weights;

        float *a;
        float *b;

        float *buffer;
        bool *buffer_full;
        unsigned int buffer_size;
    };

    using infer_kernel_fct = void (*)(KernelData &, const Spike *, bool, void *);
    static constexpr char infer_kernel_symbol[] = "infer_kernel";


    // Callback declarations
    CALLBACK float on_pre(const Spike &pre_spike, float weight);
}

extern "C" unsigned int n_synaptic_traces;
extern "C" unsigned int n_neuron_traces;

extern "C" void infer_kernel(EvSpikeSim::KernelData &kernel_data, const EvSpikeSim::Spike *end_pre_spikes,
                             bool first_call, void *thread_pool_ptr);