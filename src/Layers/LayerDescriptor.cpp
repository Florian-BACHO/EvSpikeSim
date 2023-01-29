//
// Created by Florian Bacho on 22/01/23.
//

#include "Layers/LayerDescriptor.h"

using namespace EvSpikeSim;

LayerDescriptor::LayerDescriptor(unsigned int n_inputs, unsigned int n_neurons, float tau_s, float threshold)
: n_inputs(n_inputs), n_neurons(n_neurons), tau_s(tau_s), tau(2.0f * tau_s), threshold(threshold), c(threshold / tau) {}

bool LayerDescriptor::operator==(const LayerDescriptor &rhs) const {
    return n_inputs == rhs.n_inputs &&
           n_neurons == rhs.n_neurons &&
           tau_s == rhs.tau_s &&
           tau == rhs.tau &&
           threshold == rhs.threshold &&
           c == rhs.c;
}
