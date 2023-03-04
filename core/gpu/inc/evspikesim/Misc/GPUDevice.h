//
// Created by Florian Bacho on 17/02/23.
//

#pragma once

namespace EvSpikeSim {
    /**
     * Gets the maximum number of threads per block of the current GPU.
     * @return The maximum number of threads per block of the current GPU.
     */
    unsigned int get_n_thread_per_block();
}