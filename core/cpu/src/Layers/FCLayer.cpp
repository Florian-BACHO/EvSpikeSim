//
// Created by Florian Bacho on 22/01/23.
//

#include <evspikesim/Layers/FCLayer.h>

using namespace EvSpikeSim;

FCLayer::FCLayer(unsigned int n_inputs,
                 unsigned int n_neurons,
                 float tau_s,
                 float threshold,
                 Initializer &initializer,
                 unsigned int buffer_size,
                 kernel_signature kernel) :
        Layer({n_neurons, n_inputs}, n_inputs, n_neurons, tau_s, threshold, initializer, buffer_size),
        kernel(kernel) {}

const SpikeArray &FCLayer::infer(const SpikeArray &pre_spikes) {
    FCKernelData kernel_data = {
            n_inputs,
            n_neurons,
            tau_s,
            tau,
            threshold,

            current_pre_spike,
            pre_spikes.end(),
            n_spikes,

            weights,
            a,
            b,

            buffer,
            &buffer_full,
            buffer_size
    };
    bool first_call = true;

    reset(pre_spikes);

    if (pre_spikes.is_empty())
        return post_spikes;

    do {
        reset_buffer();

        kernel(thread_pool, kernel_data, first_call);

        process_buffer();
        first_call = false;
    } while (buffer_full);

    post_spikes.sort();
    return post_spikes;
}