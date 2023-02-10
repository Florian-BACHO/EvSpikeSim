//
// Created by Florian Bacho on 22/01/23.
//

#include <cmath>
#include <evspikesim/Layers/FCLayer.h>

using namespace EvSpikeSim;

FCLayer::FCLayer(const descriptor_type &desc, std::shared_ptr<ThreadPool> &thread_pool, unsigned int buffer_size) :
        Layer(desc, thread_pool, {desc.n_neurons, desc.n_inputs}, buffer_size) {}


FCLayer::FCLayer(const descriptor_type &desc, std::shared_ptr<ThreadPool> &thread_pool, Initializer &initializer,
                 unsigned int buffer_size) :
        Layer(desc, thread_pool, {desc.n_neurons, desc.n_inputs}, initializer, buffer_size) {}

FCLayer::FCLayer(const descriptor_type &desc, std::shared_ptr<ThreadPool> &thread_pool, Initializer &&initializer,
                 unsigned int buffer_size) :
                 FCLayer(desc, thread_pool, initializer, buffer_size) {}

static inline float compute_inside_x(float a, float b, float c) {
    return b * b - 4 * a * c;
}

static inline float compute_inside_log(float a, float b, float x) {
    return 2.0f * a / (b + x);
}

static inline float compute_spike_time(float inside_log, float tau) {
    return tau * std::log(inside_log);
}

static inline float apply_reset(float b, float c, float inside_log) {
    return b - c * inside_log;
}

void FCLayer::apply_decay_and_weight(unsigned int neuron_idx, float delta_t, unsigned int pre_idx) {
    float weight = weights.get(neuron_idx, pre_idx);
    float exp_tau = exp(-delta_t / desc.tau);
    float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

    a[neuron_idx] = a[neuron_idx] * exp_tau_s + weight;
    b[neuron_idx] = b[neuron_idx] * exp_tau + weight;
}

static inline float get_next_time(const SpikeArray::const_iterator &current,
                                  const SpikeArray::const_iterator &end) {
    return ((current + 1) == end) ? (INFINITY) : ((current + 1)->time - current->time);
}

bool FCLayer::fire(unsigned int neuron_idx, float current_time, float next_pre_time, unsigned int &n_spike_buffer) {
    bool valid_spike;
    float x, inside_log, spike_time;
    float a = this->a[neuron_idx];
    float &b = this->b[neuron_idx];
    float c = desc.c;
    float tau = desc.tau;
    float last_time = 0.0;

    while (n_spike_buffer < buffer_size) {
        // Compute spike time
        x = compute_inside_x(a, b, c);
        if (x < 0)
            return false;
        x = sqrt(x);
        inside_log = compute_inside_log(a, b, x);
        if (inside_log <= 0)
            return false;
        spike_time = compute_spike_time(inside_log, tau);

        // Check for valid spike
        valid_spike = last_time < spike_time && spike_time < next_pre_time;
        if (!valid_spike)
            return false;

        // Valid spike
        last_time = spike_time;
        n_spikes[neuron_idx]++;
        buffer[neuron_idx * buffer_size + n_spike_buffer] = current_time + spike_time;
        b = apply_reset(b, c, inside_log);
        n_spike_buffer++;

        // Reached the end of the buffer
        if (n_spike_buffer == buffer_size) {
            buffer_full = true;
            return true;
        }
    }
    return false;
}

void FCLayer::infer_neuron(const SpikeArray &pre_spikes, unsigned int neuron_idx, bool first_call) {
    float next_time = 0.0;
    unsigned int n_spike_buffer = 0;
    SpikeArray::const_iterator &spike = current_pre_spike[neuron_idx];

    // Carry on inference if buffer was full in previous call
    if (!first_call && spike != pre_spikes.end()) {
        next_time = get_next_time(spike, pre_spikes.end());
        if (fire(neuron_idx, spike->time, next_time, n_spike_buffer))
            return;
        spike++;
    }
    while (spike != pre_spikes.end()) {
        apply_decay_and_weight(neuron_idx, next_time, spike->index);
        next_time = get_next_time(spike, pre_spikes.end());
        if (fire(neuron_idx, spike->time, next_time, n_spike_buffer))
            return;
        spike++;
    }
}

void FCLayer::infer_range(const SpikeArray &pre_spikes, unsigned int neuron_start, unsigned int neuron_end,
                          bool first_call) {
    for (unsigned int i = neuron_start; i < neuron_end && i < desc.n_neurons; i++)
        infer_neuron(pre_spikes, i, first_call);
}

const SpikeArray &FCLayer::infer(const SpikeArray &pre_spikes) {
    unsigned int n_neurons_per_thread = std::max(desc.n_neurons / thread_pool->get_thread_count(), 1u);
    std::vector<std::future<void>> tasks;
    bool first_call = true;

    reset(pre_spikes);

    if (pre_spikes.is_empty())
        return post_spikes;

    do {
        tasks.clear();
        reset_buffer();

        for (auto i = 0u; i < desc.n_neurons; i += n_neurons_per_thread)
            tasks.push_back(thread_pool->submit([this, &pre_spikes, i, n_neurons_per_thread, first_call] {
                infer_range(pre_spikes, i, i + n_neurons_per_thread, first_call);
            }));

        for (auto &it : tasks)
            it.get();
        process_buffer();
        first_call = false;
    } while (buffer_full);

    post_spikes.sort();
    return post_spikes;
}