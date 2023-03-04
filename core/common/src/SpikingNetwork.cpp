//
// Created by Florian Bacho on 08/02/23.
//

#include <evspikesim/SpikingNetwork.h>

using namespace EvSpikeSim;

SpikingNetwork::SpikingNetwork(const std::string &compile_path) :
        layers(), compiler(std::make_unique<JITCompiler>(compile_path)) {}

SpikingNetwork::~SpikingNetwork() {
    layers.clear(); // Unsure that layers are deleted before the JITCompiler that stores the dynamic libraries containing Layers' deleters
    compiler = nullptr;
}

const SpikeArray &SpikingNetwork::infer(const SpikeArray &pre_spikes) {
    auto *spikes = &pre_spikes;

    if (!pre_spikes.is_sorted())
        throw std::runtime_error("Input spikes must be sorted in time. Please call the .sort() method "
                                 "before inference.");
    for (auto &layer : layers)
        spikes = &layer->infer(*spikes);

    return *spikes;
}

SpikingNetwork::iterator SpikingNetwork::begin() {
    return layers.begin();
}

SpikingNetwork::iterator SpikingNetwork::end() {
    return layers.end();
}

std::shared_ptr<Layer> SpikingNetwork::get_output_layer() {
    return layers.back();
}

unsigned int SpikingNetwork::get_n_layers() const {
    return layers.size();
}