#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fc_layer.h"

static void initialize_weights(float *weights, int n_inputs, int n_neurons, float (*init_fct)()) {
    for (int i = 0; i < n_neurons * n_inputs; i++)
	weights[i] = (*init_fct)();
}

fc_layer_t *fc_layer_new(fc_layer_params_t params, float (*init_fct)()) {
    fc_layer_t *out = malloc(sizeof(fc_layer_t));

    if (out == 0)
	return 0;
    out->params = params;
    out->weights = malloc(params.n_neurons * params.n_inputs * sizeof(float));
    out->a = calloc(params.n_neurons, sizeof(float));
    out->b = calloc(params.n_neurons, sizeof(float));
    out->n_spikes = calloc(params.n_neurons, sizeof(unsigned int));
    out->total_n_spikes = 0;
    out->post_spikes = 0;
    if (out->weights == 0 || out->a == 0 || out->b == 0 || out->n_spikes == 0)
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
    spike_list_destroy(layer->post_spikes);
    free(layer);
}

void fc_layer_reset(fc_layer_t *layer) {
    memset(layer->a, 0, layer->params.n_neurons * sizeof(float));
    memset(layer->b, 0, layer->params.n_neurons * sizeof(float));
    memset(layer->n_spikes, 0, layer->params.n_neurons * sizeof(unsigned int));
    layer->total_n_spikes = 0;
    spike_list_destroy(layer->post_spikes);
}

void fc_layer_set_weights(fc_layer_t *layer, const float *new_weights) {
    size_t size = layer->params.n_inputs * layer->params.n_neurons * sizeof(float);

    memcpy(layer->weights, new_weights, size);
}

static void update_state(fc_layer_t *layer, const spike_list_t *pre_spike) {
    unsigned int pre_idx = pre_spike->index;
    float weight;

    for (unsigned int i = 0; i < layer->params.n_neurons; i++) {
	weight = layer->weights[i * layer->params.n_inputs + pre_idx];
	layer->a[i] += weight; // Time of pre-spike is 0 so just integrate the weight;
	layer->b[i] += weight;
    }
}

// Use exponential decay to avoid floating point overflow
static void apply_decay(fc_layer_t *layer, float delta_t) {
    float exp_tau = exp(-delta_t / layer->params.tau);
    float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

    for (unsigned int i = 0; i < layer->params.n_neurons; i++) {
	layer->a[i] *= exp_tau_s;
	layer->b[i] *= exp_tau;
    }
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

static bool fire_neuron(fc_layer_t *layer, int neuron_idx, float current_time, float next_pre_time) {
    float a = layer->a[neuron_idx];
    float b = layer->b[neuron_idx];
    float c = layer->params.c;
    float tau = layer->params.tau;
    float x, inside_log, spike_time, valid_spike;
    float prev_time = 0;

    do {
	x = compute_inside_x(a, b, c);
	if (x < 0)
	    break;
	x = sqrt(x);
	inside_log = compute_inside_log(a, b, x);
	if (inside_log <= 0)
	    break;
	spike_time = compute_spike_time(inside_log, tau);
	valid_spike = prev_time < spike_time && spike_time < next_pre_time;
	if (valid_spike) { // Neuron fires
	    layer->post_spikes = spike_list_add(layer->post_spikes, neuron_idx, spike_time + current_time); // Add current_time as spike time is relative to last pre-spike
	    if (layer->post_spikes == 0)
		return false; // Dynamic allocation failed
	    layer->n_spikes[neuron_idx]++;
	    layer->total_n_spikes++;
	    prev_time = spike_time;
	    b = apply_reset(b, c, inside_log);
	}
    } while (valid_spike);
    layer->b[neuron_idx] = b;
    return true;
}

static bool integrate_pre_spike(fc_layer_t *layer, const spike_list_t *pre_spike,
				float next_pre_time) {
    
    update_state(layer, pre_spike);
    for (unsigned int i = 0; i < layer->params.n_neurons; i++)
	if (!fire_neuron(layer, i, pre_spike->time, next_pre_time))
	    return false; // Dynamic allocation failed
    return true;
}

const spike_list_t *fc_layer_infer(fc_layer_t *layer, const spike_list_t *pre_spikes_start) {
    const spike_list_t *current_pre_spike = pre_spikes_start;
    float next_pre_time, delta_t;

    if (spike_list_empty(pre_spikes_start))
	return 0; // No spikes
    do {
	if (current_pre_spike != pre_spikes_start) {
	    delta_t = current_pre_spike->time - current_pre_spike->prev->time;
	    apply_decay(layer, delta_t); // Apply decay to avoid exponential overflow
	}
	next_pre_time = (current_pre_spike->next == pre_spikes_start) ? (INFINITY) :
	    (current_pre_spike->next->time - current_pre_spike->time); // Time relative to the current spike
	if (!integrate_pre_spike(layer, current_pre_spike, next_pre_time))
	    return 0; // Dynamic allocation failed
	current_pre_spike = current_pre_spike->next;
    } while (current_pre_spike != pre_spikes_start);
    return layer->post_spikes;
}
