//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/SpikingNetwork.h>

using namespace EvSpikeSim;

// Mock Initialization Functions
class IncrementalInitFct : public Initializer {
public:
    IncrementalInitFct() : counter(0.0) {}

    inline float operator()() override { return counter++; }

private:
    float counter;
};

TEST(FCLayerTest, Construction) {
    FCLayerDescriptor desc(42, 21, 0.1, 1.0);
    SpikingNetwork network = SpikingNetwork();
    auto layer = network.add_layer(desc);
    auto weights_n_dims = layer->get_weights().get_n_dims();
    const auto &weights_dims = layer->get_weights().get_dims();

    ASSERT_EQ(layer->get_descriptor(), desc);
    ASSERT_EQ(weights_n_dims, 2u);
    ASSERT_EQ(weights_dims[0], 21u);
    ASSERT_EQ(weights_dims[1], 42u);
}

TEST(FCLayerTest, ConstructionInitIncremental) {
    FCLayerDescriptor desc(42, 21, 0.1, 1.0);
    SpikingNetwork network = SpikingNetwork();
    auto layer = network.add_layer(desc, IncrementalInitFct());

    auto weights_n_dims = layer->get_weights().get_n_dims();
    const auto &weights_dims = layer->get_weights().get_dims();
    const auto &weights = layer->get_weights();

    ASSERT_EQ(layer->get_descriptor(), desc);
    ASSERT_EQ(weights_n_dims, 2u);
    ASSERT_EQ(weights_dims[0], 21u);
    ASSERT_EQ(weights_dims[1], 42u);
    float i = 0.0;
    for (auto y = 0u; y < weights_dims[0]; y++)
        for (auto x = 0u; x < weights_dims[1]; x++)
            EXPECT_FLOAT_EQ(weights.get(y, x), i++);
}

TEST(FCLayerTest, Inference) {
    FCLayerDescriptor desc(2, 3, 0.020, 0.020 * 0.2);
    SpikingNetwork network = SpikingNetwork();
    auto layer = network.add_layer(desc);
    SpikeArray input_spikes = SpikeArray();
    std::vector<float> weights = {1.0, 0.2,
                       -0.1, 0.8,
                       0.5, 0.4};
    std::vector<unsigned int> true_n_spikes = {3, 4, 3};

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

    layer->get_weights() = weights;

    input_spikes.add(0, 1.0);
    input_spikes.add(1, 1.5);
    input_spikes.add(1, 1.2);
    input_spikes.sort();

    const auto &post_spikes = layer->infer(input_spikes);

    ASSERT_EQ(post_spikes, true_outputs);
}

TEST(FCLayerTest, InferenceUndersizedBuffer) {
    FCLayerDescriptor desc(2, 3, 0.020, 0.020 * 0.2);
    SpikingNetwork network = SpikingNetwork();
    unsigned int buffer_size = 1;
    auto layer = network.add_layer(desc, buffer_size);
    SpikeArray input_spikes = SpikeArray();
    std::vector<float> weights = {1.0, 0.2,
                                  -0.1, 0.8,
                                  0.5, 0.4};
    std::vector<unsigned int> true_n_spikes = {3, 4, 3};

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

    layer->get_weights() = weights;

    input_spikes.add(0, 1.0);
    input_spikes.add(1, 1.5);
    input_spikes.add(1, 1.2);
    input_spikes.sort();

    const auto &post_spikes = layer->infer(input_spikes);

    ASSERT_EQ(post_spikes, true_outputs);
}