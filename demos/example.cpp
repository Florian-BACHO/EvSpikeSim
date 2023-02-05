//
// Created by Florian Bacho on 02/02/23.
//

#include <evspikesim/SpikingNetwork.h>
#include <evspikesim/Layers/FCLayer.h>

namespace sim = EvSpikeSim;

int main() {
    // Create network
    auto network = sim::SpikingNetwork();

    // Layer parameters
    unsigned int n_inputs = 2;
    unsigned int n_neurons = 3;
    float tau_s = 0.020;
    float threshold = tau_s * 0.2;

    // Add fully-connected layer to the network
    auto desc = sim::FCLayerDescriptor(n_inputs, n_neurons, tau_s, threshold);
    std::shared_ptr<sim::FCLayer> layer = network.add_layer(desc);

    // Set weights
    std::vector<float> weights = {1.0, 0.3,
                                  -0.1, 0.8,
                                  0.5, 0.4};
    std::copy(weights.data(), weights.data() + weights.size(), layer->get_weights().c_ptr());

    // Mutate weight
    layer->get_weights().get(0, 1) -= 0.1;

    // Create input spikes
    std::vector<unsigned int> input_indices = {0, 1, 1};
    std::vector<float> input_times = {1.0, 1.5, 1.2};

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