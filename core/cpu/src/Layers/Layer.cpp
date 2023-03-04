//
// Created by Florian Bacho on 17/02/23.
//

#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Misc/ThreadPool.h>

using namespace EvSpikeSim;

void *Layer::get_thread_pool_ptr() const {
    return static_cast<void *>(&thread_pool);
}