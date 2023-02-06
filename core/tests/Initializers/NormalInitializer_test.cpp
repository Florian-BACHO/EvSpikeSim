//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include <evspikesim/Initializers/NormalInitializer.h>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/SpikingNetwork.h>

using namespace EvSpikeSim;

TEST(NormalInitializerTest, Default) {
    std::random_device generator;
    NormalInitializer init(generator);
    float last_value = -42.0;
    float value;

    for (auto i = 0; i < 100; i++) {
        value = init();
        EXPECT_NE(value, last_value);
        last_value = value;
    }
}

TEST(NormalInitializerTest, WithArguments) {
    std::random_device generator;
    NormalInitializer init(generator, 1.0f, 42.0f);
    float last_value = -420.0;
    float value;

    for (auto i = 0; i < 100; i++) {
        value = init();
        EXPECT_NE(value, last_value);
        last_value = value;
    }
}

TEST(NormalInitializerTest, LayerInitialization) {
    std::random_device generator;
    FCLayerDescriptor desc(10, 10, 0.020, 0.020 * 0.2);
    SpikingNetwork network = SpikingNetwork();
    NormalInitializer init(generator);
    float last_value = -420.0;
    float value;

    auto layer = network.add_layer(desc, init);

    for (auto i = 0; i < 100; i++) {
        value = layer->get_weights().get_values()[i];
        EXPECT_NE(value, last_value);
        last_value = value;
    }
}