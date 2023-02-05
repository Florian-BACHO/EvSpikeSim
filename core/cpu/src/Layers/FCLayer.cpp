//
// Created by Florian Bacho on 22/01/23.
//

#include <cmath>
#include <evspikesim/Layers/FCLayer.h>

using namespace EvSpikeSim;

void FCLayer::process_buffer() {
    float time;

    for (unsigned int i = 0; i < desc.n_neurons; i++) {
        for (unsigned int j = 0; j < buffer_size; j++) {
            time = buffer[i * buffer_size + j];
            if (time == infinity)
                break;
            post_spikes.add(i, time);
        }
    }
}

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

void FCLayer::fire(unsigned int neuron_idx, float current_time, float next_pre_time, unsigned int &n_spike_buffer) {
    bool valid_spike;
    float x, inside_log, spike_time;
    float a = this->a[neuron_idx];
    float &b = this->b[neuron_idx];
    float c = desc.c;
    float tau = desc.tau;
    float last_time = 0.0;

    while (n_spike_buffer < buffer_size) {
        x = compute_inside_x(a, b, c);
        if (x < 0)
            return;
        x = sqrt(x);
        inside_log = compute_inside_log(a, b, x);
        if (inside_log <= 0)
            return;
        spike_time = compute_spike_time(inside_log, tau);
        valid_spike = last_time < spike_time && spike_time < next_pre_time;
        if (!valid_spike)
            return;
        last_time = spike_time;
        n_spikes[neuron_idx]++;
        buffer[neuron_idx * buffer_size + n_spike_buffer] = current_time + spike_time;
        b = apply_reset(b, c, inside_log);
        n_spike_buffer++;
    }
}

void FCLayer::infer_neuron(const SpikeArray &pre_spikes, unsigned int neuron_idx) {
    float next_time = 0.0;
    unsigned int n_spike_buffer = 0;

    for (auto it = pre_spikes.begin(); it != pre_spikes.end(); it++) {
        apply_decay_and_weight(neuron_idx, next_time, it->index);
        next_time = get_next_time(it, pre_spikes.end());
        fire(neuron_idx, it->time, next_time, n_spike_buffer);
    }
}

void FCLayer::infer_range(const SpikeArray &pre_spikes, unsigned int neuron_start, unsigned int neuron_end) {
    for (unsigned int i = neuron_start; i < neuron_end && i < desc.n_neurons; i++)
        infer_neuron(pre_spikes, i);
}

const SpikeArray &FCLayer::infer(const SpikeArray &pre_spikes) {
    unsigned int n_neurons_per_thread = std::max(desc.n_neurons / thread_pool->get_thread_count(), 1u);
    std::vector<std::future<void>> tasks;

    if (pre_spikes.empty())
        return post_spikes;

    reset();
    for (auto i = 0u; i < desc.n_neurons; i += n_neurons_per_thread)
        tasks.push_back(thread_pool->submit([this, pre_spikes, i, n_neurons_per_thread] {
            infer_range(pre_spikes, i, i + n_neurons_per_thread);
        }));

    for (auto &it : tasks)
        it.get();
    process_buffer();

    post_spikes.sort();
    return post_spikes;
}