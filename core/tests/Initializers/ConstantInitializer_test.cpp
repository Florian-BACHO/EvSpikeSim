//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include <evspikesim/Initializers/ConstantInitializer.h>
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/SpikingNetwork.h>

using namespace EvSpikeSim;

TEST(ConstantInitializerTest, Default) {
    ConstantInitializer init;

    for (auto i = 0; i < 100; i++)
        EXPECT_EQ(init(), 0.0f);
}

TEST(ConstantInitializerTest, Value) {
    ConstantInitializer init(42.21f);

    for (auto i = 0; i < 100; i++)
        EXPECT_EQ(init(), 42.21f);
}

TEST(ConstantInitializerTest, LayerInitialization) {
    SpikingNetwork network = SpikingNetwork();
    ConstantInitializer init(42.21f);

    auto layer = network.add_layer<FCLayer>(10, 10, 0.020, 0.020, init);

    for (auto i = 0; i < 100; i++)
        EXPECT_EQ(layer->get_weights().get_values()[i], 42.21f);
}