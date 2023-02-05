//
// Created by Florian Bacho on 21/01/23.
//

#pragma once

#include <iostream>
#include <vector>
#include <iterator>
#include <evspikesim/Spike.h>

namespace EvSpikeSim {
    class SpikeArray {
    public:
        using const_iterator = std::vector<Spike>::const_iterator;

    public:
        SpikeArray();

        SpikeArray(const std::vector<unsigned int> &indices, const std::vector<float> &times);

        template<class IndexIterator, class TimeIterator>
        SpikeArray(IndexIterator begin_indices, IndexIterator end_indices, TimeIterator begin_times) :
                spikes(), sorted(true) {
            add(begin_indices, end_indices, begin_times);
        }

        void add(unsigned int index, float time);

        void add(const std::vector<unsigned int> &indices, const std::vector<float> &times);

        template<class IndexIterator, class TimeIterator>
        void add(IndexIterator begin_indices, IndexIterator end_indices, TimeIterator begin_times) {
            while (begin_indices != end_indices) {
                add(*begin_indices, *begin_times);
                begin_indices++;
                begin_times++;
            }
        }

        void sort();

        void clear();

        const_iterator begin() const;

        const_iterator end() const;

        std::size_t n_spikes() const;

        bool is_sorted() const;

        bool empty() const;

        const Spike *c_ptr() const;

        bool operator==(const SpikeArray &rhs) const;

        bool operator!=(const SpikeArray &rhs) const;

    private:
        static constexpr size_t block_size = 1024;

        std::vector<Spike> spikes;
        bool sorted;

    private:
        inline bool is_max_capacity() const;

        inline void extend_capacity();
    };

    // IOStreams
    std::ostream &operator<<(std::ostream &os, const EvSpikeSim::SpikeArray &array);
}