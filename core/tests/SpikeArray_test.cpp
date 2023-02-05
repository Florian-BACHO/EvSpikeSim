//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include <evspikesim/Spike.h>
#include <evspikesim/SpikeArray.h>

using namespace EvSpikeSim;

TEST(SpikeArrayTest, NSpikesAndEmpty) {
    SpikeArray arr;

    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.n_spikes(), 0u);

    arr.add(21, 42.42);
    arr.add(12, 1.42);
    arr.add(12, 21.84);

    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.n_spikes(), 3u);
}

TEST(SpikeArrayTest, AddAndIteration) {
    SpikeArray arr;

    arr.add(21, 42.42);
    arr.add(12, 1.42);
    arr.add(12, 21.84);

    auto it = arr.begin();

    EXPECT_EQ(*it, Spike(21, 42.42));
    EXPECT_EQ(*(++it), Spike(12, 1.42));
    EXPECT_EQ(*(++it), Spike(12, 21.84));
    EXPECT_EQ(++it, arr.end());
}

TEST(SpikeArrayTest, AddWithIterators) {
    std::vector<unsigned int> indices = {21, 12, 12};
    std::vector<float> times = {42.42, 1.42, 21.84};
    SpikeArray arr(indices.begin(), indices.end(), times.begin());


    auto it = arr.begin();

    EXPECT_EQ(*it, Spike(21, 42.42));
    EXPECT_EQ(*(++it), Spike(12, 1.42));
    EXPECT_EQ(*(++it), Spike(12, 21.84));
    EXPECT_EQ(++it, arr.end());

    SpikeArray arr2;

    arr2.add(indices.begin(), indices.end(), times.begin());

    it = arr2.begin();

    EXPECT_EQ(*it, Spike(21, 42.42));
    EXPECT_EQ(*(++it), Spike(12, 1.42));
    EXPECT_EQ(*(++it), Spike(12, 21.84));
    EXPECT_EQ(++it, arr2.end());
}

TEST(SpikeArrayTest, AddWithVectors) {
    std::vector<unsigned int> indices = {21, 12, 12};
    std::vector<float> times = {42.42, 1.42, 21.84};
    SpikeArray arr(indices, times);


    auto it = arr.begin();

    EXPECT_EQ(*it, Spike(21, 42.42));
    EXPECT_EQ(*(++it), Spike(12, 1.42));
    EXPECT_EQ(*(++it), Spike(12, 21.84));
    EXPECT_EQ(++it, arr.end());

    SpikeArray arr2;

    arr2.add(indices, times);

    it = arr2.begin();

    EXPECT_EQ(*it, Spike(21, 42.42));
    EXPECT_EQ(*(++it), Spike(12, 1.42));
    EXPECT_EQ(*(++it), Spike(12, 21.84));
    EXPECT_EQ(++it, arr2.end());
}

TEST(SpikeArrayTest, AddWithCPointers) {
    unsigned int indices[] = {21, 12, 12};
    float times[] = {42.42, 1.42, 21.84};
    SpikeArray arr(indices, indices + 3, times);


    auto it = arr.begin();

    EXPECT_EQ(*it, Spike(21, 42.42));
    EXPECT_EQ(*(++it), Spike(12, 1.42));
    EXPECT_EQ(*(++it), Spike(12, 21.84));
    EXPECT_EQ(++it, arr.end());

    SpikeArray arr2;

    arr2.add(indices, indices + 3, times);

    it = arr2.begin();

    EXPECT_EQ(*it, Spike(21, 42.42));
    EXPECT_EQ(*(++it), Spike(12, 1.42));
    EXPECT_EQ(*(++it), Spike(12, 21.84));
    EXPECT_EQ(++it, arr2.end());
}


TEST(SpikeArrayTest, Sort) {
    SpikeArray arr;

    arr.add(21, 42.42);
    arr.add(12, 1.42);
    arr.add(12, 21.84);

    arr.sort();

    auto it = arr.begin();

    EXPECT_EQ(*(it), Spike(12, 1.42));
    EXPECT_EQ(*(++it), Spike(12, 21.84));
    EXPECT_EQ(*(++it), Spike(21, 42.42));
    EXPECT_EQ(++it, arr.end());
}

TEST(SpikeArrayTest, Clear) {
    SpikeArray arr;

    arr.add(21, 42.42);
    arr.add(12, 1.42);
    arr.add(12, 21.84);


    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.n_spikes(), 3u);

    arr.clear();

    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.n_spikes(), 0u);
}

TEST(SpikeArrayTest, Comparators) {
    SpikeArray arr1;
    SpikeArray arr2;

    EXPECT_TRUE(arr1 == arr2);
    EXPECT_FALSE(arr1 != arr2);

    arr1.add(21, 42.42);
    arr1.add(12, 1.42);
    arr1.add(12, 21.84);
    arr1.sort();


    EXPECT_FALSE(arr1 == arr2);
    EXPECT_TRUE(arr1 != arr2);

    arr2.add(21, 42.42);

    EXPECT_FALSE(arr1 == arr2);
    EXPECT_TRUE(arr1 != arr2);

    arr2.add(12, 21.84);
    arr2.add(12, 1.42);

    EXPECT_FALSE(arr1 == arr2);
    EXPECT_TRUE(arr1 != arr2);

    arr2.sort();

    EXPECT_TRUE(arr1 == arr2);
    EXPECT_FALSE(arr1 != arr2);
}

TEST(SpikeArrayTest, OStream) {
    SpikeArray arr;

    arr.add(21, 42.42);
    arr.add(12, 1.42);
    arr.add(12, 21.84);

    arr.sort();

    std::stringstream ss;

    ss << arr;
    EXPECT_STREQ(ss.str().c_str(), "Index: 12, Time: 1.42\n"
                                   "Index: 12, Time: 21.84\n"
                                   "Index: 21, Time: 42.42\n");
}