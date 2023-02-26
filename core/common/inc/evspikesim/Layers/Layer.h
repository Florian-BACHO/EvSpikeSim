//
// Created by Florian Bacho on 22/01/23.
//

#pragma once

#include <memory>
#include <limits>
#include <initializer_list>
#include <evspikesim/SpikeArray.h>
#include <evspikesim/Layers/InferKernelDeclarations.h>
#include <evspikesim/Misc/NDArray.h>
#include <evspikesim/Misc/ContainerTypes.h>

namespace EvSpikeSim {
    /**
     * Base class for layers.
     */
    class Layer {
    public:
        /**
         * Constructs the base layer.
         * @param weights_dims The dimension of weights.
         * @param n_inputs The number of input neurons.
         * @param n_neurons The number of neurons in the layer.
         * @param tau_s The synaptic time constant.
         * @param threshold The threshold.
         * @param initializer The initializer to use to initialize weights.
         * @param buffer_size The size (per neuron) of the post-synaptic spike times buffer used during inference.
         * @param traces_tau_fct The function returning the time constants of synaptic and neuron eligibility traces.
         * @param kernel_fct The inference kernel function.
         */
        Layer(const std::initializer_list<unsigned int> &weights_dims,
              unsigned int n_inputs,
              unsigned int n_neurons,
              float tau_s,
              float threshold,
              Initializer &initializer,
              unsigned int buffer_size,
              get_traces_tau_fct traces_tau_fct = get_traces_tau,
              infer_kernel_fct kernel_fct = infer_kernel);

        /**
         * Infer the layer using the given pre-synaptic spike array.
         * @param pre_spikes Pre-synaptic spikes.
         * @return Constant reference on the post-synaptic spike array.
         */
        const SpikeArray &infer(const SpikeArray &pre_spikes);

        /**
         * Gets the number of input neurons.
         * @return The number of input neurons.
         */
        inline unsigned int get_n_inputs() const { return n_inputs; }

        /**
         * Gets the number of neurons in the layer.
         * @return The number of neurons in the layer.
         */
        inline unsigned int get_n_neurons() const { return n_neurons; }

        /**
         * Gets the synaptic time constant.
         * @return The synaptic time constant.
         */
        inline float get_tau_s() const { return tau_s; }

        /**
         * Gets the membrane time constant.
         * @return The membrane time constant (2 * tau_s).
         */
        inline float get_tau() const { return tau; }

        /**
         * Gets the threshold.
         * @return The threshold.
         */
        inline float get_threshold() const { return threshold; }

        /**
         * Gets the weights of the layer
         * @return A reference on the NDArray storing the weights of the layer.
         */
        inline NDArray<float> &get_weights() { return weights; };

        /**
         * Gets the post-synaptic spikes (updated after calling infer).
         * @return A const reference on the post-synaptic spike array
         */
        inline const SpikeArray &get_post_spikes() const { return post_spikes; };

        /**
         * Gets the number of post-synaptic spikes fired by each neuron.
         * @return A array of spike counts.
         */
        inline const EvSpikeSim::vector<unsigned int> &get_n_spikes() const { return n_spikes; }

        /**
         * Gets the number of eligibility traces per synapse.
         * @return The number of eligibility traces per synapse.
         */
        inline unsigned int get_n_synaptic_traces() const { return static_cast<unsigned int>(synaptic_traces_tau.size()); }

        /**
         * Gets the number of traces per neuron (to not be confused with synaptic traces).
         * @return The number of traces per neuron.
         */
        inline unsigned int get_n_neuron_traces() const { return static_cast<unsigned int>(neuron_traces_tau.size()); }

        /**
         * Gets all the synaptic eligibility traces.
         * @return All the synaptic eligibility traces.
         */
        inline const EvSpikeSim::vector<float> &get_synaptic_traces() const { return synaptic_traces; };

        /**
         * Gets all the neuron eligibility traces (to not be confused with synaptic traces).
         * @return All the neuron eligibility traces.
         */
        inline const EvSpikeSim::vector<float> &get_neuron_traces() const { return neuron_traces; };

    protected:
        /**
         * Resets the layer, including membrane potential and traces.
         * @param pre_spikes
         */
        void reset(const SpikeArray &pre_spikes);

        /**
         * Resets the post-synaptic spike time buffer but not the neurons. This is used during inference if the
         * post-synaptic spike buffer becomes full.
         */
        void reset_buffer();

        /**
         * Processes the post-synaptic spike buffer into the post-synaptic spike array.
         */
        void process_buffer();

    protected:
        static constexpr float infinity = std::numeric_limits<float>::infinity(); /**< The float value of infinity */

        const unsigned int n_inputs; /**< The number of pre-synaptic neurons. */
        const unsigned int n_neurons; /**< The number of post-synaptic neurons. */
        const float tau_s; /**< The synaptic time constant. */
        const float tau; /**< The membrane time constant (2 * tau_s). */
        const float threshold; /**< The threshold. */

        // Outputs
        SpikeArray post_spikes; /**< The post-synaptic spike array */
        EvSpikeSim::vector<unsigned int> n_spikes; /**< The post-synaptic spike counts */

        // Connectionns
        NDArray<float> weights; /**< The weights of the layer */

        // Inference states
        EvSpikeSim::vector<const Spike *> current_pre_spike; /**< Keeps track of current pre-synaptic spikes of each neuron during inference */
        EvSpikeSim::vector<float> current_time; /**< Keeps track of current simulation time for each neuron during inference */
        EvSpikeSim::vector<float> a; /**< State a of each neuron: Sum w * exp(t/tau_s) */
        EvSpikeSim::vector<float> b; /**< State b of each neuron: Sum w * exp(t/tau) - reset */

        // Buffer
        EvSpikeSim::vector<float> buffer; /**< The post-synaptic spike times buffer. */
        unsigned int buffer_size; /**< The size (per neuron) of the post-synaptic spike times buffer. */
        EvSpikeSim::unique_ptr<bool> buffer_full; /**< Indicates when the post-synaptic spike times buffer is full. */

        // Traces
        EvSpikeSim::vector<float> synaptic_traces_tau; /**< Time constants of synaptic traces. */
        EvSpikeSim::vector<float> neuron_traces_tau; /**< Time constants of neuron traces. */
        EvSpikeSim::vector<float> synaptic_traces;  /**< Synaptic eligibility traces. */
        EvSpikeSim::vector<float> neuron_traces; /**< Neuron eligibility traces. */

        // Inference kernel
        KernelData kernel_data; /**< A convenient structure that stores all pointers and data required for the inference. */
        infer_kernel_fct kernel_fct; /**< The inference kernel function used for inference. If compiled from a custom
                                      * source file, this points to the kernel function in the loaded dynamic library.
                                      * Otherwise, the default infer_kernel is used.*/

    private:
        /**
         * Initializes eligibility traces.
         * @param traces_tau_fct The function returning the time constants of synaptic and neuron eligibility traces.
         */
        void init_traces(get_traces_tau_fct traces_tau_fct);

        /**
         * Creates the kernel data structure that is passed to kernel_fct during for inference.
         * @return A structure containing all the data required for inference.
         */
        KernelData get_kernel_data();

        /**
         * Gets a void * pointer on the global thread pool.
         * @return A pointer on the global thread pool in the CPU implementation.
         * If EvSpikeSim is compiled for GPUs, the function returns nullptr as the thread pool is unused.
         */
        void *get_thread_pool_ptr() const; // Return nullptr in GPU implementation
    };
}