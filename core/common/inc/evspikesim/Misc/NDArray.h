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

namespace EvSpikeSim {
    template<typename T = float, class Allocator = std::allocator<T>>
    class NDArray {

    public:
        NDArray(const std::initializer_list<unsigned int> &dimensions) :
                dims(dimensions), values(NDArray::count_n_elems(dimensions)) {}

        NDArray(const std::initializer_list<unsigned int> &dimensions, Initializer &fct) :
                NDArray(dimensions) {
            for (auto &it : values)
                it = fct();
        }

        NDArray(const std::initializer_list<unsigned int> &dimensions, Initializer &&fct) :
                NDArray(dimensions) {
            for (auto &it : values)
                it = fct();
        }

        NDArray(const std::initializer_list<unsigned int> &dimensions, std::shared_ptr<Initializer> &fct) :
                NDArray(dimensions) {
            for (auto &it : values)
                it = (*fct)();
        }

        unsigned int get_n_dims() const {
            return dims.size();
        }

        std::vector<unsigned int> get_dims() const {
            return dims;
        }

        std::size_t size() const {
            return values.size();
        }

        template<typename U, typename... Args>
        T &get(U first_idx, Args... indices) {
            return values[get_index(dims.begin(), U(0), first_idx, indices...)];
        }

        std::vector<T, Allocator> &get_values() {
            return values;
        }

        template <class IteratorType>
        void set_values(const IteratorType &begin, const IteratorType &end) {
            values.assign(begin, end);
        }

        template <class ContainerType>
        void set_values(const ContainerType &other) {
            set_values(other.begin(), other.end());
        }

        template<typename U, typename... Args>
        const T &get(U first_idx, Args... indices) const {
            return values[get_index(dims.begin(), U(0), first_idx, indices...)];
        }

        template<typename U, typename... Args>
        void set(T value, U first_idx, Args... indices) {
            values[get_index(dims.begin(), U(0), first_idx, indices...)] = value;
        }

        T *get_c_ptr() {
            return values.data();
        }

        template <class ContainerType>
        NDArray<T, Allocator> &operator=(const ContainerType &other) {
            set_values(other);
            return *this;
        }

    private:
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

        template<typename Iterator, typename U>
        U get_index(Iterator dim, U out) const {
            (void) dim; // Unused

            return out;
        }

        template<typename Iterator, typename U, typename... Args>
        U get_index(Iterator dim, U out, U idx, Args... indices) const {
            return get_index(dim + 1, out * U(*dim) + idx, indices...);
        }

    private:
        const std::vector<unsigned int> dims;
        std::vector<T, Allocator> values;
    };
}