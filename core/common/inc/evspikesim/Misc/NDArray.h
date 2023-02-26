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
         * Constructs an uninitialized array with the given dimensions.
         * @param dimensions Initializer list containing the dimensions of the array.
         */
        NDArray(const std::initializer_list<unsigned int> &dimensions) :
                dims(dimensions), values(NDArray::count_n_elems(dimensions)) {}

        /**
         * Constructs an array with the given dimensions and initializes it with the given initializer
         * @param dimensions Initializer list containing the dimensions of the array.
         * @param fct Initializer that initializes the values of the array.
         */
        NDArray(const std::initializer_list<unsigned int> &dimensions, Initializer &fct) :
                NDArray(dimensions) {
            for (auto &it : values)
                it = fct();
        }

        /**
         * Constructs an array with the given dimensions and initializes it with the given initializer
         * @param dimensions Initializer list containing the dimensions of the array.
         * @param fct Initializer that initializes the values of the array.
         */
        NDArray(const std::initializer_list<unsigned int> &dimensions, Initializer &&fct) :
                NDArray(dimensions) {
            for (auto &it : values)
                it = fct();
        }

        /**
         * Constructs an array with the given dimensions and initializes it with the given initializer
         * @param dimensions Initializer list containing the dimensions of the array.
         * @param fct Initializer that initializes the values of the array.
         */
        NDArray(const std::initializer_list<unsigned int> &dimensions, std::shared_ptr<Initializer> &fct) :
                NDArray(dimensions) {
            for (auto &it : values)
                it = (*fct)();
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
        T &get(U first_idx, Args... indices) {
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
        const T &get(U first_idx, Args... indices) const {
            return values[get_index(dims.begin(), U(0), first_idx, indices...)];
        }

        /**
         * Set a new value at the specified index.  Usage: arr.get(-4, 3, 2, 4) sets the new value -4 in a 3D array.
         * @tparam U Type of the first index.
         * @tparam Args Types of the remaining indices.
         * @param value The new value to set.
         * @param first_idx The first index.
         * @param indices The remaining indices.
         */
        template<typename U, typename... Args>
        void set(T value, U first_idx, Args... indices) {
            values[get_index(dims.begin(), U(0), first_idx, indices...)] = value;
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
         * Counts the number of element in the dimension initializer list.
         * @param dimensions The dimensions of the newly created array.
         * @return The number of elements in the initializer list.
         */
        static std::size_t count_n_elems(const std::initializer_list<unsigned int> &dimensions) {
            auto it = dimensions.begin();
            std::size_t n_elems;

            if (it == dimensions.end())
                return 0;
            n_elems = *it;
            while (++it != dimensions.end())
                n_elems *= *it;
            return n_elems;
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
        const std::vector<unsigned int> dims; /**< The dimensions of the array */
        EvSpikeSim::vector<T> values; /**< The data of the array */
    };
}