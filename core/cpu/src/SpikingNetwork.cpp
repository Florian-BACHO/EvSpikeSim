//
// Created by Florian Bacho on 23/01/23.
//

#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/SpikingNetwork.h>

using namespace EvSpikeSim;


std::shared_ptr<FCLayer> SpikingNetwork::add_layer(const FCLayerDescriptor &descriptor,
                                                   unsigned int buffer_size) {
    auto layer = std::make_shared<FCLayer>(descriptor, thread_pool, buffer_size);

    layers.push_back(layer);
    return layer;
}

std::shared_ptr<FCLayer> SpikingNetwork::add_layer(const FCLayerDescriptor &descriptor, Initializer &initializer,
                                                   unsigned int buffer_size) {
    auto layer = std::make_shared<FCLayer>(descriptor, thread_pool, initializer, buffer_size);

    layers.push_back(layer);
    return layer;
}

std::shared_ptr<FCLayer> SpikingNetwork::add_layer(const FCLayerDescriptor &descriptor, Initializer &&initializer,
                                                   unsigned int buffer_size) {
    return add_layer(descriptor, initializer, buffer_size);
}