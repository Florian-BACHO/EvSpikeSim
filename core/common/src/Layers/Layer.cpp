//
// Created by Florian Bacho on 08/02/23.
//

#include <evspikesim/Layers/Layer.h>

using namespace EvSpikeSim;

void Layer::process_buffer() {
    float time;

    for (unsigned int i = 0; i < desc.n_neurons; i++) {
        for (unsigned int j = 0; j < buffer_size; j++) {
            time = buffer[i * buffer_size + j];
            if (time == infinity)
                break;
            post_spikes.add(i, time);
        }
    }
}