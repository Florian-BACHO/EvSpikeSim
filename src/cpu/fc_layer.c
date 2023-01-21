#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fc_layer.h"

static void initialize_weights(float *weights, int n_inputs, int n_neurons,
			       float (*init_fct)(void)) {
    for (int i = 0; i < n_neurons * n_inputs; i++)
	weights[i] = (*init_fct)();
}

fc_layer_t *fc_layer_new(fc_layer_params_t params, float (*init_fct)(void),
			 infer_thread_pool_t *thread_pool) {
    fc_layer_t *out = malloc(sizeof(fc_layer_t));

    if (out == 0)
	return 0;
    out->params = params;
    out->weights = malloc(params.n_neurons * params.n_inputs * sizeof(float));
    out->a = calloc(params.n_neurons, sizeof(float));
    out->b = calloc(params.n_neurons, sizeof(float));
    out->thread_pool = thread_pool;
    out->buffer_full = malloc(sizeof(bool));
    out->spike_time_buffer = malloc(params.n_neurons * BUFFER_SIZE * sizeof(float));
    out->n_spikes = calloc(params.n_neurons, sizeof(unsigned int));
    out->total_n_spikes = 0;
    out->post_spikes = 0;
    if (out->weights == 0 || out->a == 0 || out->b == 0 || out->n_spikes == 0 ||
	out->buffer_full == 0 || out->spike_time_buffer == 0)
	return 0;
    if (init_fct != 0)
	initialize_weights(out->weights, params.n_inputs, params.n_neurons, init_fct);
    return out;
}

void fc_layer_destroy(fc_layer_t *layer) {
    free(layer->weights);
    free(layer->a);
    free(layer->b);
    free(layer->n_spikes);
    free(layer->buffer_full);
    free(layer->spike_time_buffer);
    spike_list_destroy(layer->post_spikes);
    free(layer);
}

void fc_layer_reset(fc_layer_t *layer) {
    memset(layer->a, 0, layer->params.n_neurons * sizeof(float));
    memset(layer->b, 0, layer->params.n_neurons * sizeof(float));
    memset(layer->n_spikes, 0, layer->params.n_neurons * sizeof(unsigned int));
    layer->total_n_spikes = 0;
    spike_list_destroy(layer->post_spikes);
    layer->post_spikes = 0;
}

void fc_layer_set_weights(fc_layer_t *layer, const float *new_weights) {
    size_t size = layer->params.n_inputs * layer->params.n_neurons * sizeof(float);

    memcpy(layer->weights, new_weights, size);
}

static bool process_buffer_spikes(fc_layer_t *layer) {
    float time;

    for (unsigned int i = 0; i < layer->params.n_neurons; i++) {
        for (unsigned int j = 0; j < BUFFER_SIZE; j++) {
            time = layer->spike_time_buffer[i * BUFFER_SIZE + j];
            if (time == INFINITY)
                break;
            layer->post_spikes = spike_list_add(layer->post_spikes, i, time);
            layer->total_n_spikes++;
            if (layer->post_spikes == 0)
                return false;
	}
    }
    return true;
}

static inline float compute_inside_x(float a, float b, float c) {
    return b * b - 4 * a * c;
}

static inline float compute_inside_log(float a, float b, float x) {
    return 2.0f * a / (b + x);
}

static inline float compute_spike_time(float inside_log, float tau) {
    return tau * log(inside_log);
}

static inline float apply_reset(float b, float c, float inside_log) {
    return b - c * inside_log;
}

static inline float get_next_time(const spike_list_t *current_pre_spike,
				  const spike_list_t *pre_spikes_start) {
    return (current_pre_spike->next == pre_spikes_start) ? (INFINITY) :
	(current_pre_spike->next->time - current_pre_spike->time);
}

static void apply_decay_and_weight(fc_layer_t *layer, unsigned int neuron_idx, float delta_t,
				   unsigned int pre_idx) {
    float weight = layer->weights[neuron_idx * layer->params.n_inputs + pre_idx];
    float exp_tau = exp(-delta_t / layer->params.tau);
    float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

    layer->a[neuron_idx] = layer->a[neuron_idx] * exp_tau_s + weight;
    layer->b[neuron_idx] = layer->b[neuron_idx] * exp_tau + weight;
}

static void fire(fc_layer_t *layer, unsigned int neuron_idx, float current_time,
		 float next_pre_time, unsigned int *n_spike_buffer) {
    bool valid_spike;
    float x, inside_log, spike_time;
    float a = layer->a[neuron_idx];
    float *b = layer->b + neuron_idx;
    float c = layer->params.c;
    float tau = layer->params.tau;
    float last_time = 0.0;

    while (*n_spike_buffer < BUFFER_SIZE) {
	x = compute_inside_x(a, *b, c);
        if (x < 0)
	    return;
        x = sqrt(x);
        inside_log = compute_inside_log(a, *b, x);
        if (inside_log <= 0)
	    return;
        spike_time = compute_spike_time(inside_log, tau);
        valid_spike = last_time < spike_time && spike_time < next_pre_time;
        if (!valid_spike)
	    return;
        last_time = spike_time;
        layer->n_spikes[neuron_idx]++;
	layer->spike_time_buffer[neuron_idx * BUFFER_SIZE +
				 *n_spike_buffer] = current_time + spike_time;
	*b = apply_reset(*b, c, inside_log);
	(*n_spike_buffer)++;
    }
    *(layer->buffer_full) = true;
}

static void infer_neuron(fc_layer_t *layer, const spike_list_t *pre_spikes_start,
			 unsigned int neuron_idx) {
    float next_time = 0.0;
    unsigned int n_spike_buffer = 0;
    const spike_list_t *current_pre_spike = pre_spikes_start;

    if (spike_list_empty(pre_spikes_start))
	return;
    do {
	apply_decay_and_weight(layer, neuron_idx, next_time, current_pre_spike->index);
	next_time = get_next_time(current_pre_spike, pre_spikes_start);
	fire(layer, neuron_idx, current_pre_spike->time, next_time, &n_spike_buffer);
	current_pre_spike = current_pre_spike->next;
    } while (current_pre_spike != pre_spikes_start);
}

static void infer_neuron_range(fc_layer_t *layer, const spike_list_t *pre_spikes_start,
			       unsigned int neuron_start, unsigned int neuron_end) {
    for (unsigned int i = neuron_start; i < neuron_end && i < layer->params.n_neurons; i++)
	infer_neuron(layer, pre_spikes_start, i);
}

const void init_spike_time_buffer(float *buffer, unsigned int n_neurons) {
    for (unsigned int i = 0; i < n_neurons * BUFFER_SIZE; i++)
	buffer[i] = INFINITY;
}

const spike_list_t *fc_layer_infer(fc_layer_t *layer, const spike_list_t *pre_spikes_start,
				   unsigned int total_n_spikes) {
    if (spike_list_empty(pre_spikes_start))
	return 0; // No spikes
    init_spike_time_buffer(layer->spike_time_buffer, layer->params.n_neurons);
    if (layer->thread_pool == 0) {
	for (unsigned int i = 0; i < layer->params.n_neurons; i++)
	    infer_neuron(layer, pre_spikes_start, i);
    } else {
	layer->buffer_full = false;
	for (unsigned int i = 0; i < layer->params.n_neurons; i += layer->thread_pool->n_threads)
	    infer_thread_pool_add_work(layer->thread_pool, (infer_fct_t)&infer_neuron_range, layer,
				       pre_spikes_start, i, i + layer->thread_pool->n_threads);
	infer_thread_pool_wait(layer->thread_pool);
    }
    if (!process_buffer_spikes(layer))
	return 0;
    // do {
    // while (*(layer->buffer_full));
    return layer->post_spikes;
}
