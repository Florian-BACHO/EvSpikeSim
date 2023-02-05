//
// Created by Florian Bacho on 21/01/23.
//

#pragma once

#include <iostream>
#include <vector>
#include <iterator>
#include <evspikesim/Spike.h>
#include <evspikesim/Misc/CudaManagedAllocator.h>

/*
 * Notes on the Spike vector container for SpikeArray and the efficiency of spike sorting.
 *
 * We conduced experiments on sorting of std::vector<float>, std::vector<float, CudaManagedAllocator<float>>,
 * thrust::universal_vector<float> and thrust::device_vector<float>
 *
 * For 1000 elements and 1000 iterations:
 *
 * std::vector<float>, std::sort: 218ms
 * thrust::universal_vector<float>, std::sort: 5028ms
 * thrust::universal_vector<float>, thrust::sort: 5022ms
 * std::vector<float, cudaManagedAllocator<float>>, std::sort: 45ms (best)
 * std::vector<float, cudaManagedAllocator<float>>, thrust::sort: 111ms
 * thrust::device_vector<float>, thrust::sort: 117ms
 *
 * For 100 000 elements and 10 iterations:
 *
 * std::vector<float>, std::sort: 154ms
 * thrust::universal_vector<float>, std::sort: 4968ms
 * thrust::universal_vector<float>, thrust::sort: 4910ms
 * std::vector<float, cudaManagedAllocator<float>>, std::sort: 69ms
 * std::vector<float, cudaManagedAllocator<float>>, thrust::sort: 16ms
 * thrust::device_vector<float>, thrust::sort: 12ms (best)
 *
 * thrust::universal_vector (unified memory) are very slow as they seem to transfer data every time an element is
 * pushed on the cpu.
 * Thrust::sort is efficient only on large vectors (10x faster than CPU with 100 000 elements) but std::sort seems to
 * be faster on GPU
 */

namespace EvSpikeSim {
    class SpikeArray {
    public:
        using const_iterator = std::vector<Spike, CudaManagedAllocator<Spike>>::const_iterator;

    public:
        SpikeArray() = default;

        SpikeArray(const std::vector<unsigned int> &indices, const std::vector<float> &times);

        template<class IndexIterator, class TimeIterator>
        SpikeArray(IndexIterator begin_indices, IndexIterator end_indices, TimeIterator begin_times) {
            add(begin_indices, end_indices, begin_times);
        }

        void add(unsigned int index, float time);
        void add(const std::vector<unsigned int> &indices, const std::vector<float> &times);

        template<class IndexIterator, class TimeIterator>
        void add(IndexIterator begin_indices, IndexIterator end_indices, TimeIterator begin_times) {
            while (begin_indices != end_indices) {
                if (is_max_capacity())
                    extend_capacity();
                spikes.emplace_back(*begin_indices, *begin_times);
                begin_indices++;
                begin_times++;
            }
        }

        void sort();

        void clear();

        inline const_iterator begin() const { return spikes.begin(); };

        inline const_iterator end() const { return spikes.end(); };

        inline std::size_t n_spikes() const { return spikes.size(); }

        inline bool empty() const { return n_spikes() == 0; }

        inline const Spike *c_ptr() const { return spikes.data(); }

        bool operator==(const SpikeArray &rhs) const;

        bool operator!=(const SpikeArray &rhs) const;

    private:
        static constexpr size_t block_size = 1024;

        std::vector<Spike, CudaManagedAllocator<Spike>> spikes;

    private:
        inline bool is_max_capacity() const { return spikes.size() == spikes.capacity(); }

        inline void extend_capacity() { spikes.reserve(spikes.capacity() + block_size); }
    };

    // IOStreams
    std::ostream &operator<<(std::ostream &os, const EvSpikeSim::SpikeArray &array);
}