//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <initializer_list>
#include <functional>
#include <array>
#include <vector>

namespace EvSpikeSim {
    template <typename T = float, class Allocator = std::allocator<T>>
    class NDArray {
    public:
        using init_fct = std::function<T()>;

    public:
        NDArray(const std::initializer_list<unsigned int> &dimensions) : dims(dimensions),
        values(NDArray::count_n_elems(dimensions)) {}

        NDArray(const std::initializer_list<unsigned int> &dimensions, T fill_value) : dims(dimensions),
        values(NDArray::count_n_elems(dimensions), fill_value) {}

        NDArray(const std::initializer_list<unsigned int> &dimensions, const init_fct &fct) : NDArray(dimensions) {
            for (auto &it : values)
                it = fct();
        }

        inline unsigned int get_n_dims() const {
            return dims.size();
        }

        inline std::vector<unsigned int> get_dims() const {
            return dims;
        }

        inline std::size_t size() const {
            return values.size();
        }

        template <typename U, typename... Args>
        inline T &get(U first_idx, Args... indices) {
            return values[get_index(dims.begin(), U(0), first_idx, indices...)];
        }

        template <typename U, typename... Args>
        inline const T &get(U first_idx, Args... indices) const {
            return values[get_index(dims.begin(), U(0), first_idx, indices...)];
        }

        template <typename U, typename... Args>
        inline void set(T value, U first_idx, Args... indices) {
            values[get_index(dims.begin(), U(0), first_idx, indices...)] = value;
        }

        inline T *c_ptr() {
            return values.data();
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

        template <typename Iterator, typename U>
        U get_index(Iterator dim, U out) const {
            (void)dim; // Unused

            return out;
        }

        template <typename Iterator, typename U, typename... Args>
        U get_index(Iterator dim, U out, U idx, Args... indices) const {
            return get_index(dim + 1, out * U(*dim) + idx, indices...);
        }

    private:
        const std::vector<unsigned int> dims;
        std::vector<T, Allocator> values;
    };
}