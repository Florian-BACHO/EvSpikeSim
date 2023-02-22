//
// Created by Florian Bacho on 17/02/23.
//

#include <evspikesim/Misc/GPUDevice.h>

unsigned int EvSpikeSim::get_n_thread_per_block() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}