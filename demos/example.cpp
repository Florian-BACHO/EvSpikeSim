//
// Created by Florian Bacho on 02/02/23.
//

#include <evspikesim/SpikingNetwork.h>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/Initializers/UniformInitializer.h>
#include <evspikesim/Misc/RandomGenerator.h>

namespace sim = EvSpikeSim;

int main() {
    // Create network
    sim::SpikingNetwork network;

    // Layer parameters
    unsigned int n_inputs = 3;
    unsigned int n_neurons = 30;
    float tau_s = 0.020;
    float threshold = 0.1;

    // Uniform initial distribution (by default: [-1, 1])
    sim::RandomGenerator generator;
    sim::UniformInitializer init(generator);

    // Add fully-connected layer to the network
    std::shared_ptr<sim::FCLayer> layer = network.add_layer<sim::FCLayer>(n_inputs, n_neurons, tau_s, threshold, init);

    // Input spikes
    std::vector<unsigned int> input_indices = {0, 1, 2, 1};
    std::vector<float> input_times = {1.0, 1.5, 1.2, 1.1};

    // Inference
    auto output_spikes = network.infer(input_indices, input_times);

    std::cout << "Output spikes:" << std::endl;
    std::cout << output_spikes << std::endl;

    std::cout << "Output spike counts:" << std::endl;
    for (auto it : layer->get_n_spikes())
        std::cout << it << " ";
    std::cout << std::endl;
    return 0;
}