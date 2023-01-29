//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include "Initializers/UniformInitializer.h"

using namespace EvSpikeSim;

TEST(UniformInitializerTest, Default) {
    std::random_device generator;
    UniformInitializer init(generator);
    float last_value = -42.0;
    float value;

    for (auto i = 0; i < 100; i++) {
        value = init();
        EXPECT_GE(value, -1.0f);
        EXPECT_LT(value, 1.0f);
        EXPECT_NE(value, last_value);
        last_value = value;
    }
}

TEST(UniformInitializerTest, Bounds) {
    std::random_device generator;
    UniformInitializer init(generator, -42.0f, 42.0f);
    float last_value = -420.0;
    float value;

    for (auto i = 0; i < 100; i++) {
        value = init();
        EXPECT_GE(value, -42.0f);
        EXPECT_LT(value, 42.0f);
        EXPECT_NE(value, last_value);
        last_value = value;
    }
}