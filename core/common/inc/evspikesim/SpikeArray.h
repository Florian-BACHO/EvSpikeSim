//
// Created by Florian Bacho on 21/01/23.
//

#pragma once

#include <iostream>
#include <vector>
#include <iterator>
#include <evspikesim/Spike.h>
#include <evspikesim/Misc/ContainerTypes.h>

namespace EvSpikeSim {
    /**
     * An array of Spike. SpikeArray objects must be sorted in time before being used for inference.
     */
    class SpikeArray {
    public:
        using const_iterator = EvSpikeSim::vector<Spike>::const_iterator; /**< Type definition of const iterator of Spike objects*/

    public:
        /**
         * Constructs an empty array of spikes.
         */
        SpikeArray();

        /**
         * Constructs an array of spikes with the given spike indices and spike timings.
         * @param indices Indices of spikes.
         * @param times Timings of spikes.
         */
        SpikeArray(const std::vector<unsigned int> &indices, const std::vector<float> &times);

        /**
         * Constructs an array of spikes with the given spike index and spike timing iterators (number of indices
         * and timings must match).
         * @tparam IndexIterator Type of index iterator.
         * @tparam TimeIterator Type of time iterator.
         * @param begin_indices First spike index.
         * @param end_indices Last spike index.
         * @param begin_times First spike timing.
         */
        template<class IndexIterator, class TimeIterator>
        SpikeArray(IndexIterator begin_indices, IndexIterator end_indices, TimeIterator begin_times) :
                spikes(), sorted(true) {
            add(begin_indices, end_indices, begin_times);
        }

        /**
         * Adds a single spike to the array.
         * @param index Index of the neuron.
         * @param time Timing of the spike.
         */
        void add(unsigned int index, float time);

        /**
         * Adds several spikes to the array given indices and timings.
         * @param indices Neuron indices that fired the spikes.
         * @param times Timings of the spikes.
         */
        void add(const std::vector<unsigned int> &indices, const std::vector<float> &times);

        /**
         * Adds spikes with the given spike index and spike timing iterators (number of indices
         * and timings must match).
         * @tparam IndexIterator Type of index iterator.
         * @tparam TimeIterator Type of time iterator.
         * @param begin_indices First spike index.
         * @param end_indices Last spike index.
         * @param begin_times First spike timing.
         */
        template<class IndexIterator, class TimeIterator>
        void add(IndexIterator begin_indices, IndexIterator end_indices, TimeIterator begin_times) {
            while (begin_indices != end_indices) {
                add(*begin_indices, *begin_times);
                begin_indices++;
                begin_times++;
            }
        }

        /**
         * Sorts the spike array in time. Must be called before being used for inference.
         */
        void sort();

        /**
         * Empty the spike array.
         */
        void clear();

        /**
         * Gets a constant iterator on the first spike.
         * @return A constant iterator on the first spike.
         */
        const_iterator begin() const;

        /**
         * Gets a constant iterator on the end of the spike array.
         * @return A constant iterator on the end of the spike array.
         */
        const_iterator end() const;

        /**
         * Gets the number of spikes in the array.
         * @return The size of the array.
         */
        std::size_t n_spikes() const;

        /**
         * Checks if the spike array is sorted in time.
         * @return True if the array is sorted in time.
         */
        bool is_sorted() const;

        /**
         * Checks if the spike array is empty.
         * @return True if the array is empty.
         */
        bool is_empty() const;

        /**
         * Gets the raw pointer of the spike array.
         * @return The raw pointer of the spike array.
         */
        const Spike *get_c_ptr() const;

        /**
         * Checks the equality betweentwo spike arrays.
         * @param rhs The other spike array to compare.
         * @return True if the two arrays have the same number of spikes and if all the spikes are equals.
         */
        bool operator==(const SpikeArray &rhs) const;

        /**
         * Checks the equality between two spike arrays.
         * @param rhs The other spike array to compare.
         * @return True if the two arrays do not have the same number of spikes or if at least one spike is not equal to the corresponding one in rhs.
         */
        bool operator!=(const SpikeArray &rhs) const;

        /**
         * Formats the spike array to the given output stream.
         * @param os The output stream to write.
         * @param array The array of spikes to format.
         * @return The output stream os.
         */
        // IOStreams
        friend std::ostream &operator<<(std::ostream &os, const EvSpikeSim::SpikeArray &array);

    private:
        static constexpr size_t block_size = 1024; /**< Size of allocation when reserve memory for spikes. */

        EvSpikeSim::vector<Spike> spikes; /**< Vector that contains spikes. */
        bool sorted; /**< Indicates if the array is sorted. */

    private:
        /**
         * Checks if the vector cannot contain any more spike.
         * @return True if
         */
        inline bool is_max_capacity() const;

        inline void extend_capacity();
    };

    std::ostream &operator<<(std::ostream &os, const EvSpikeSim::SpikeArray &array);
}