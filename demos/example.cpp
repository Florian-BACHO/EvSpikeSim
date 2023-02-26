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
    unsigned int n_inputs = 10;
    unsigned int n_neurons = 100;
    float tau_s = 0.010;
    float threshold = 0.1;

    // Uniform distribution for weight initialization (by default: [-1, 1])
    sim::RandomGenerator gen;
    sim::UniformInitializer init(gen);

    // Add fully-connected layer to the network
    std::shared_ptr<sim::FCLayer> layer = network.add_layer<sim::FCLayer>(n_inputs, n_neurons, tau_s, threshold, init);

    // Create input spikes
    std::vector<unsigned int> input_indices = {0, 8, 2, 4};
    std::vector<float> input_times = {0.010, 0.012, 0.21, 0.17};

    // Inference
    auto output_spikes = network.infer(input_indices, input_times);

    // Print output spikes
    std::cout << "Output spikes:" << std::endl;
    std::cout << output_spikes << std::endl;

    // Print output spike counts
    std::cout << "Output spike counts:" << std::endl;
    for (auto it : layer->get_n_spikes())
        std::cout << it << " ";
    std::cout << std::endl;
    return 0;
}