//
// Created by Florian Bacho on 23/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/SpikingNetwork.h>

using namespace EvSpikeSim;

TEST(SpikingNetworkTest, AddLayer) {
    FCLayerDescriptor desc(2, 3, 0.020, 0.020 * 0.2);
    SpikingNetwork network = SpikingNetwork();

    ASSERT_EQ(network.get_n_layers(), 0u);

    network.add_layer(desc);

    ASSERT_EQ(network.get_n_layers(), 1u);

    network.add_layer(desc);
    network.add_layer(desc);

    ASSERT_EQ(network.get_n_layers(), 3u);
}

TEST(SpikingNetworkTest, Inference) {
    FCLayerDescriptor desc(2, 3, 0.020, 0.020 * 0.2);
    SpikingNetwork network = SpikingNetwork();
    auto layer = network.add_layer(desc);

    SpikeArray input_spikes = SpikeArray();
    float weights[] = {1.0, 0.2,
                       -0.1, 0.8,
                       0.5, 0.4};

    SpikeArray true_outputs = SpikeArray();
    true_outputs.add(0, 1.0047829);
    true_outputs.add(0, 1.0112512);
    true_outputs.add(0, 1.0215546);
    true_outputs.add(1, 1.2063813);
    true_outputs.add(1, 1.2163547);
    true_outputs.add(1, 1.506313);
    true_outputs.add(1, 1.5162327);
    true_outputs.add(2, 1.0129402);
    true_outputs.add(2, 1.2233235);
    true_outputs.add(2, 1.5267321);
    true_outputs.sort();

    std::copy(weights, weights + 6, layer->get_weights().c_ptr());

    input_spikes.add(0, 1.0);
    input_spikes.add(1, 1.5);
    input_spikes.add(1, 1.2);
    input_spikes.sort();

    const auto &post_spikes = network.infer(input_spikes);

    ASSERT_EQ(post_spikes, true_outputs);
}

TEST(SpikingNetworkTest, ResetInference) {
    FCLayerDescriptor desc(2, 3, 0.020, 0.020 * 0.2);
    SpikingNetwork network = SpikingNetwork();
    auto layer = network.add_layer(desc);

    SpikeArray input_spikes = SpikeArray();
    float weights[] = {1.0, 0.2,
                       -0.1, 0.8,
                       0.5, 0.4};

    SpikeArray true_outputs = SpikeArray();
    true_outputs.add(0, 1.0047829);
    true_outputs.add(0, 1.0112512);
    true_outputs.add(0, 1.0215546);
    true_outputs.add(1, 1.2063813);
    true_outputs.add(1, 1.2163547);
    true_outputs.add(1, 1.506313);
    true_outputs.add(1, 1.5162327);
    true_outputs.add(2, 1.0129402);
    true_outputs.add(2, 1.2233235);
    true_outputs.add(2, 1.5267321);
    true_outputs.sort();

    std::copy(weights, weights + 6, layer->get_weights().c_ptr());

    input_spikes.add(0, 1.0);
    input_spikes.add(1, 1.5);
    input_spikes.add(1, 1.2);
    input_spikes.sort();

    network.infer(input_spikes); // Run first time

    const auto &post_spikes = network.infer(input_spikes);

    ASSERT_EQ(post_spikes, true_outputs);
}

TEST(SpikingNetworkTest, Accessor) {
    FCLayerDescriptor desc(2, 3, 0.020, 0.020 * 0.2);
    SpikingNetwork network = SpikingNetwork();
    auto layer_1 = network.add_layer(desc);
    auto layer_2 = network.add_layer(desc);
    auto layer_3 = network.add_layer(desc);

    ASSERT_EQ(layer_1, network[0]);
    ASSERT_NE(layer_1, network[1]);
    ASSERT_NE(layer_1, network[2]);

    ASSERT_NE(layer_2, network[0]);
    ASSERT_EQ(layer_2, network[1]);
    ASSERT_NE(layer_2, network[2]);

    ASSERT_NE(layer_3, network[0]);
    ASSERT_NE(layer_3, network[1]);
    ASSERT_EQ(layer_3, network[2]);
}

TEST(SpikingNetworkTest, Iterator) {
    FCLayerDescriptor desc(2, 3, 0.020, 0.020 * 0.2);
    SpikingNetwork network = SpikingNetwork();
    std::vector<std::shared_ptr<Layer>> layers = {network.add_layer(desc),
                                                  network.add_layer(desc),
                                                  network.add_layer(desc)};
    auto layer_begin = layers.begin();

    for (auto it : network) {
        ASSERT_EQ(it, *layer_begin);
        layer_begin++;
    }
}

TEST(SpikingNetworkTest, GetOutputLayer) {
    FCLayerDescriptor desc(2, 3, 0.020, 0.020 * 0.2);
    FCLayerDescriptor desc2(3, 2, 0.030, 0.030 * 0.2);
    SpikingNetwork network = SpikingNetwork();

    network.add_layer(desc);
    auto true_output = network.add_layer(desc2);

    ASSERT_EQ(network.get_output_layer(), true_output);
}