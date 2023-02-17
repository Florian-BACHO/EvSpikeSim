//
// Created by Florian Bacho on 16/02/23.
//

// Trigger kernel source definition when include FCLayerKernel.h
#ifndef KERNEL_SOURCE_DEFINITION
#define KERNEL_SOURCE_DEFINITION
#endif

#include <evspikesim/Layers/FCLayerKernel.h>

CALLBACK float EvSpikeSim::on_pre(const EvSpikeSim::Spike &pre_spike, float weight) {
    (void)pre_spike;

    return weight;
}