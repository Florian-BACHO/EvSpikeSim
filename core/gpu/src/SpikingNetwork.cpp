//
// Created by Florian Bacho on 23/01/23.
//

#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/SpikingNetwork.h>

using namespace EvSpikeSim;

SpikingNetwork::SpikingNetwork(const std::string &compile_path) :
        layers(), compiler(std::make_unique<JITCompiler>(compile_path)) {}

SpikingNetwork::~SpikingNetwork() {
    layers.clear(); // Unsure that layers are deleted before the JITCompiler that stores the dynamic libraries containing Layers' deleters
    compiler = nullptr;
}