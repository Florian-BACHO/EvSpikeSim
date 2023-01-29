//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include "Spike.h"

using namespace EvSpikeSim;

TEST(SpikeTest, Values) {
    Spike spike(42, 21.42);

    EXPECT_EQ(spike.index, 42);
    EXPECT_FLOAT_EQ(spike.time, 21.42);
}

TEST(SpikeTest, Comparators) {
    Spike spike_1(42, 21.42);
    Spike spike_2(1, 10.84);
    Spike spike_3(1, 21.42);

    // >
    EXPECT_TRUE(spike_1 > spike_2);
    EXPECT_FALSE(spike_2 > spike_1);
    EXPECT_FALSE(spike_1 > spike_3);
    EXPECT_FALSE(spike_3 > spike_1);

    // >=
    EXPECT_TRUE(spike_1 >= spike_2);
    EXPECT_FALSE(spike_2 >= spike_1);
    EXPECT_TRUE(spike_1 >= spike_3);
    EXPECT_TRUE(spike_3 >= spike_1);

    // <
    EXPECT_FALSE(spike_1 < spike_2);
    EXPECT_TRUE(spike_2 < spike_1);
    EXPECT_FALSE(spike_1 < spike_3);
    EXPECT_FALSE(spike_3 < spike_1);

    // <=
    EXPECT_FALSE(spike_1 <= spike_2);
    EXPECT_TRUE(spike_2 <= spike_1);
    EXPECT_TRUE(spike_1 <= spike_3);
    EXPECT_TRUE(spike_3 <= spike_1);

    // ==
    EXPECT_FALSE(spike_1 == spike_2);
    EXPECT_FALSE(spike_2 == spike_1);
    EXPECT_TRUE(spike_1 == spike_3);
    EXPECT_TRUE(spike_3 == spike_1);

    // !=
    EXPECT_TRUE(spike_1 != spike_2);
    EXPECT_TRUE(spike_2 != spike_1);
    EXPECT_FALSE(spike_1 != spike_3);
    EXPECT_FALSE(spike_3 != spike_1);
}

TEST(SpikeTest, OStream) {
    Spike spike(42, 21.42);
    std::stringstream ss;

    ss << spike;
    EXPECT_STREQ(ss.str().c_str(), "Index: 42, Time: 21.42");
}