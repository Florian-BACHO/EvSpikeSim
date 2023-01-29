//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include "Layers/LayerDescriptor.h"

using namespace EvSpikeSim;

TEST(LayerDescriptorTest, Values) {
    LayerDescriptor desc(42, 21, 0.1, 1.0);

    EXPECT_EQ(desc.n_inputs, 42);
    EXPECT_EQ(desc.n_neurons, 21);
    EXPECT_FLOAT_EQ(desc.tau_s, 0.1);
    EXPECT_FLOAT_EQ(desc.tau, 0.2);
    EXPECT_FLOAT_EQ(desc.threshold, 1.0);
    EXPECT_FLOAT_EQ(desc.c, 5.0);
}