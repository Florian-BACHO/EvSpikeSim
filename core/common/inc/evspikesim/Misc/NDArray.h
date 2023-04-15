//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <initializer_list>
#include <functional>
#include <array>
#include <vector>
#include <evspikesim/Initializers/Initializer.h>
#include <evspikesim/Misc/ContainerTypes.h>

namespace EvSpikeSim {
    /**
     * An array container with N dimensions.
     * @tparam T Type of values stored by the container.
     * @tparam Allocator Allocator class. std::allocator<T> for CPU NDArrays and CudaManagedAllocator<T> for GPU
     * NDArrays.
     */
    template<typename T = float>
    class NDArray {
    public:
        /**
         * Default constructor. Constructs an empty array with 0 dimensions.
         */
        NDArray() = default;

        /**
         * Constructs a new array with the given dimensions.
         * @tparam Args The types of the dimensions (variadic).
         * @param args The variadic dimension arguments.
         */
        template <typename... Args>
        NDArray(Args... args) {
            init_array(1, args...);
        }

        /**
         * Initializes the values of the array with the given initializer.
         * @param init The initializer that will initialize the values.
         */
        void initialize(Initializer &init) {
            for (auto &it : values)
                it = init();
        }

        /**
         * Initializes the values of the array with the given initializer.
         * @param init The initializer that will initialize the values.
         */
        void initialize(Initializer &&init) {
            initialize(init);
        }

        /**
         * Gets the number of dimensions of the array.
         * @return The number of dimensions of the array.
         */
        unsigned int get_n_dims() const {
            return dims.size();
        }

        /**
         * Gets the dimensions of the array.
         * @return A vector containing the size of each axis in the array.
         */
        std::vector<unsigned int> get_dims() const {
            return dims;
        }

        /**
         * Gets the total number of elements in the array.
         * @return The total number of elements in the array.
         */
        std::size_t size() const {
            return values.size();
        }

        /**
         * Gets the vector containing the values of the array.
         * @return The vector containing the values of the array.
         */
        EvSpikeSim::vector<T> &get_values() {
            return values;
        }

        /**
         * Set new values using the given iterators. The iterators should contain the right number of elements.
         * @tparam IteratorType Type of iterators.
         * @param begin Iterator on the first new element.
         * @param end Iterator on the end of the new elements.
         */
        template<class IteratorType>
        void set_values(const IteratorType &begin, const IteratorType &end) {
            values.assign(begin, end);
        }

        /**
         * Set new values using another container. Values are copied. The other container should contain the right
         * number of elements.
         * @tparam ContainerType Type of the other container.
         * @param other The other container used to set new values.
         */
        template<class ContainerType>
        void set_values(const ContainerType &other) {
            set_values(other.begin(), other.end());
        }

        /**
         * Gets the value at the specified indices. Usage: arr.get(3, 2, 4) for a 3D array.
         * @tparam U Type of the first index.
         * @tparam Args Types of the remaining indices.
         * @param first_idx The first index.
         * @param indices The remaining indices.
         * @return A reference on the value stored at the given indices.
         */
        template<typename U, typename... Args>
        T &operator()(U first_idx, Args... indices) {
            return values[get_index(dims.begin(), U(0), first_idx, indices...)];
        }

        /**
         * Gets the constant value at the specified indices. Usage: arr.get(3, 2, 4) for a 3D array.
         * @tparam U Type of the first index.
         * @tparam Args Types of the remaining indices.
         * @param first_idx The first index.
         * @param indices The remaining indices.
         * @return A const reference on the value stored at the given indices.
         */
        template<typename U, typename... Args>
        const T &operator()(U first_idx, Args... indices) const {
            return values[get_index(dims.begin(), U(0), first_idx, indices...)];
        }

        /**
         * Gets the raw pointer on the data.
         * @return The raw pointer on the data.
         */
        T *get_c_ptr() {
            return values.data();
        }

        /**
         * Set new values using another container. Values are copied. The other container should contain the right
         * number of elements.
         * @tparam ContainerType Type of the other container.
         * @param other The other container used to set new values.
         * @return Reference on the self NDArray.
         */
        template<class ContainerType>
        NDArray<T> &operator=(const ContainerType &other) {
            set_values(other);
            return *this;
        }

    private:
        /**
         * Terminal condition of variadic initialization.
         * @tparam U Type of n_elems.
         * @tparam V Type of the current dimension to process.
         * @param n_elems Number of elements accumulated.
         * @param dim The current dimension to process.
         */
        template<typename U, typename V>
        void init_array(U n_elems, V dim) {
            dims.push_back(dim);
            values = EvSpikeSim::vector<T>(n_elems * dim);
        }

        /**
         * Initializes the array with variadic dimension arguments.
         * @tparam U Type of n_elems.
         * @tparam V Type of the current dimension to process.
         * @tparam Args Types of the remaining dimensions.
         * @param n_elems Number of elements accumulated. Must be 1 at first call.
         * @param dim The current dimension to process.
         * @param args The remaining dimensions to process.
         */
        template<typename U, typename V, typename... Args>
        void init_array(U n_elems, V dim, Args... args) {
            dims.push_back(dim);
            init_array(n_elems * dim, args...);
        }

        /**
         * Gets the index in a 1D indexation. This is used as a terminal condition of the ND indexation.
         * @tparam Iterator Type of the dimension iterator.
         * @tparam U Type of the index.
         * @param dim Current dimension
         * @param out Accumulated index.
         * @return The corresponding index in the data.
         */
        template<typename Iterator, typename U>
        U get_index(Iterator dim, U out) const {
            (void) dim; // Unused

            return out;
        }

        /**
         * Recursively computes the index in a ND indexation.
         * @tparam Iterator Type of the dimension iterator.
         * @tparam U Type of the index.
         * @tparam Args Types of the remaining indices.
         * @param dim Current dimension.
         * @param out The accumulated index in the data.
         * @param idx The current unpacked index.
         * @param indices The remaining indices to process.
         * @return The corresponding index in the data.
         */
        template<typename Iterator, typename U, typename... Args>
        U get_index(Iterator dim, U out, U idx, Args... indices) const {
            return get_index(dim + 1, out * U(*dim) + idx, indices...);
        }

    private:
        std::vector<unsigned int> dims; /**< The dimensions of the array */
        EvSpikeSim::vector<T> values; /**< The data of the array */
    };
}