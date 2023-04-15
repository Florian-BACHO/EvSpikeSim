//
// Created by Florian Bacho on 22/01/23.
//

#include <string>
#include <gtest/gtest.h>
#include <evspikesim/Misc/NDArray.h>

using namespace EvSpikeSim;

// Mock Initialization Functions
class IncrementalInitFct : public Initializer {
public:
    IncrementalInitFct() : counter(0.0) {}

    float operator()() override { return counter++; }

private:
    float counter;
};

TEST(NDArrayTest, DefaultConstructor) {
    NDArray<float> arr;

    EXPECT_EQ(arr.get_n_dims(), 0u);
}

TEST(NDArrayTest, Dimensions) {
    NDArray<float> mat(21, 42);
    auto dims = mat.get_dims();

    EXPECT_EQ(mat.get_n_dims(), 2u);
    EXPECT_EQ(dims[0], 21u);
    EXPECT_EQ(dims[1], 42u);
    EXPECT_EQ(mat.size(), 21u * 42u);
}

TEST(NDArrayTest, InitializeIncremental) {
    NDArray<float> mat(21, 42);
    auto dims = mat.get_dims();

    mat.initialize(IncrementalInitFct());
    EXPECT_EQ(dims[0], 21u);
    EXPECT_EQ(dims[1], 42u);
    float i = 0.0;
    for (auto y = 0u; y < dims[0]; y++)
        for (auto x = 0u; x < dims[1]; x++)
            EXPECT_FLOAT_EQ(mat(y, x), i++);
}

TEST(NDArrayTest, SetValue) {
    NDArray<float> mat(21, 42);
    auto dims = mat.get_dims();

    mat(2, 3) = 42.21;
    EXPECT_EQ(dims[0], 21u);
    EXPECT_EQ(dims[1], 42u);
    for (auto y = 0u; y < dims[0]; y++)
        for (auto x = 0u; x < dims[1]; x++)
            EXPECT_FLOAT_EQ(mat(y, x), (y == 2 && x == 3) ? 42.21 : 0.0);
}

TEST(NDArrayTest, CPtr) {
    NDArray<float> tensor(21, 42, 84);
    float *ptr = tensor.get_c_ptr();
    auto dims = tensor.get_dims();

    EXPECT_EQ(dims[0], 21u);
    EXPECT_EQ(dims[1], 42u);
    EXPECT_EQ(dims[2], 84u);

    ptr[(2 * dims[1] + 12) * dims[2] + 3] = 42.21;
    for (auto z = 0u; z < dims[0]; z++)
        for (auto y = 0u; y < dims[1]; y++)
            for (auto x = 0u; x < dims[2]; x++)
                EXPECT_FLOAT_EQ(tensor(z, y, x), (z == 2 && y == 12 && x == 3) ? 42.21 : 0.0);
}

TEST(NDArrayTest, Size) {
    NDArray<float> tensor(21, 42, 84);

    EXPECT_EQ(tensor.size(), 74088u);
}

TEST(NDArrayTest, SetGetValues) {
    NDArray<float> tensor({2, 3});
    EvSpikeSim::vector<float> new_values = {0, 1,
                                            2, 3,
                                            4, 5};

    tensor.set_values(new_values);
    EXPECT_EQ(tensor.get_values(), new_values);
}

TEST(NDArrayTest, SetGetValuesIterator) {
    NDArray<float> tensor(2, 3);
    EvSpikeSim::vector<float> new_values = {0, 1,
                                            2, 3,
                                            4, 5};

    tensor.set_values(new_values.begin(), new_values.end());
    EXPECT_EQ(tensor.get_values(), new_values);
}

TEST(NDArrayTest, AssignOperator) {
    NDArray<float> tensor(2, 3);
    EvSpikeSim::vector<float> new_values = {0, 1,
                                            2, 3,
                                            4, 5};

    tensor = new_values;
    EXPECT_EQ(tensor.get_values(), new_values);
}
