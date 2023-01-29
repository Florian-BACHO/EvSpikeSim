//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include "Misc/NDArray.h"

using namespace EvSpikeSim;

// Mock Initialization Functions
class IncrementalInitFct {
public:
    IncrementalInitFct() : counter(0.0) {}

    inline float operator()() { return counter++; }

private:
    float counter;
};

static float constantInitFct() {
    return 42.21;
}

TEST(NDArrayTest, Dimensions) {
    NDArray<float> mat({21, 42});
    auto dims = mat.get_dims();

    EXPECT_EQ(mat.get_n_dims(), 2);
    EXPECT_EQ(dims[0], 21);
    EXPECT_EQ(dims[1], 42);
    EXPECT_EQ(mat.size(), 21 * 42);
}


TEST(NDArrayTest, FillConstructor) {
    NDArray<float> mat({21, 42}, 84.42);
    auto dims = mat.get_dims();

    EXPECT_EQ(dims[0], 21);
    EXPECT_EQ(dims[1], 42);
    for (auto y = 0u; y < dims[0]; y++)
        for (auto x = 0u; x < dims[1]; x++)
            EXPECT_FLOAT_EQ(mat.get(y, x), 84.42);
}

TEST(NDArrayTest, InitConstructorConstant) {
    NDArray<float> mat({21, 42}, constantInitFct);
    auto dims = mat.get_dims();

    EXPECT_EQ(dims[0], 21);
    EXPECT_EQ(dims[1], 42);
    for (auto y = 0u; y < dims[0]; y++)
        for (auto x = 0u; x < dims[1]; x++)
            EXPECT_FLOAT_EQ(mat.get(y, x), 42.21f);
}

TEST(NDArrayTest, InitConstructorIncremental) {
    NDArray<float> mat({21, 42}, IncrementalInitFct());
    auto dims = mat.get_dims();

    EXPECT_EQ(dims[0], 21);
    EXPECT_EQ(dims[1], 42);
    float i = 0.0;
    for (auto y = 0u; y < dims[0]; y++)
        for (auto x = 0u; x < dims[1]; x++)
            EXPECT_FLOAT_EQ(mat.get(y, x), i++);
}

TEST(NDArrayTest, SetValue) {
    NDArray<float> mat({21, 42}, 0.0);
    auto dims = mat.get_dims();

    mat.set(42.21, 2, 3);
    EXPECT_EQ(dims[0], 21);
    EXPECT_EQ(dims[1], 42);
    for (auto y = 0u; y < dims[0]; y++)
        for (auto x = 0u; x < dims[1]; x++)
            EXPECT_FLOAT_EQ(mat.get(y, x), (y == 2 && x == 3) ? 42.21 : 0.0);
}

TEST(NDArrayTest, CPtr) {
    NDArray<float> tensor({21, 42, 84}, 0.0);
    float *ptr = tensor.c_ptr();
    auto dims = tensor.get_dims();

    EXPECT_EQ(dims[0], 21);
    EXPECT_EQ(dims[1], 42);
    EXPECT_EQ(dims[2], 84);

    ptr[(2 * dims[1] + 12) * dims[2] + 3] = 42.21;
    for (auto z = 0u; z < dims[0]; z++)
        for (auto y = 0u; y < dims[1]; y++)
            for (auto x = 0u; x < dims[2]; x++)
                EXPECT_FLOAT_EQ(tensor.get(z, y, x), (z == 2 && y == 12 && x == 3) ? 42.21 : 0.0);
}

TEST(NDArrayTest, Size) {
    NDArray<float> tensor({21, 42, 84});

    EXPECT_EQ(tensor.size(), 74088);
}