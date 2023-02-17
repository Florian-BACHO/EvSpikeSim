//
// Created by Florian Bacho on 14/02/23.
//

#ifndef KERNEL_SOURCE_DEFINITION
#define KERNEL_SOURCE_DEFINITION
#endif

#include <evspikesim/Layers/FCLayerKernel.h>

CALLBACK float EvSpikeSim::on_pre(const Spike &pre_spike, float weight) {
    (void)pre_spike;

    return 2.0f * weight;
}