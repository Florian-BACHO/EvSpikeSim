//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include "Initializers/ConstantInitializer.h"

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