#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
    
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "fc_layer.h"

static void initialize_weights(float *weights, int n_inputs, int n_neurons, float (*init_fct)()) {
    for (int i = 0; i < n_neurons * n_inputs; i++)
	weights[i] = (*init_fct)();
}

fc_layer_t *fc_layer_new(fc_layer_params_t params, float (*init_fct)()) {
  fc_layer_t *out = (fc_layer_t *)malloc(sizeof(fc_layer_t));

    if (out == 0)
	return 0;
    out->params = params;
    if (cudaMallocManaged(&out->weights, params.n_neurons * params.n_inputs * sizeof(float)) ||
	cudaMallocManaged(&out->a, params.n_neurons * sizeof(float)) ||
	cudaMallocManaged(&out->b, params.n_neurons * sizeof(float)) ||
	cudaMallocManaged(&out->fired, sizeof(bool)) ||
	cudaMallocManaged(&out->spike_times, params.n_neurons * sizeof(float)))
      return 0;
    cudaMemset(out->a, 0, params.n_neurons * sizeof(float));
    cudaMemset(out->b, 0, params.n_neurons * sizeof(float));
    out->n_spikes = (unsigned int *)calloc(params.n_neurons, sizeof(unsigned int));
    if (out->n_spikes == 0)
	return 0;
    out->total_n_spikes = 0;
    out->post_spikes = 0;
    if (init_fct != 0)
	initialize_weights(out->weights, params.n_inputs, params.n_neurons, init_fct);
    return out;
}

void fc_layer_destroy(fc_layer_t *layer) {
    cudaFree(layer->weights);
    cudaFree(layer->a);
    cudaFree(layer->b);
    cudaFree(layer->fired);
    cudaFree(layer->spike_times);
    free(layer->n_spikes);
    spike_list_destroy(layer->post_spikes);
    free(layer);
}

void fc_layer_reset(fc_layer_t *layer) {
    cudaMemset(layer->a, 0, layer->params.n_neurons * sizeof(float));
    cudaMemset(layer->b, 0, layer->params.n_neurons * sizeof(float));
    memset(layer->n_spikes, 0, layer->params.n_neurons * sizeof(unsigned int));
    layer->total_n_spikes = 0;
    spike_list_destroy(layer->post_spikes);
    layer->post_spikes = 0;
}

void fc_layer_set_weights(fc_layer_t *layer, const float *new_weights) {
    size_t size = layer->params.n_inputs * layer->params.n_neurons * sizeof(float);

    cudaMemcpy(layer->weights, new_weights, size, cudaMemcpyDefault);
}

__global__ void update_state_kernel(const float *weights, float *a, float *b, unsigned int pre_idx,
				    unsigned int n_inputs) {
    unsigned int neuron_idx = threadIdx.x;
    float weight = weights[neuron_idx * n_inputs + pre_idx];

    a[neuron_idx] += weight;
    b[neuron_idx] += weight;
}

static void update_state(fc_layer_t *layer, const spike_list_t *pre_spike) {
    unsigned int pre_idx = pre_spike->index;
    unsigned int n_inputs = layer->params.n_inputs;
    unsigned int n_neurons = layer->params.n_neurons;

    update_state_kernel<<<1, n_neurons>>>(layer->weights, layer->a, layer->b, pre_idx, n_inputs);
}

__global__ void apply_decay_kernel(float *a, float *b, float exp_tau, float exp_tau_s) {
    unsigned int neuron_idx = threadIdx.x;
 
    a[neuron_idx] *= exp_tau_s;
    b[neuron_idx] *= exp_tau;
}

// Use exponential decay to avoid floating point overflow
static void apply_decay(fc_layer_t *layer, float delta_t) {
    unsigned int n_neurons = layer->params.n_neurons;
    float exp_tau = exp(-delta_t / layer->params.tau);
    float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

    apply_decay_kernel<<<1, n_neurons>>>(layer->a, layer->b, exp_tau, exp_tau_s);
}

__inline__ __device__ float compute_inside_x(float a, float b, float c) {
    return b * b - 4 * a * c;
}

__inline__ __device__ float compute_inside_log(float a, float b, float x) {
    return 2.0f * a / (b + x);
}

__inline__ __device__ float compute_spike_time(float inside_log, float tau) {
    return tau * log(inside_log);
}

__inline__ __device__ float apply_reset(float b, float c, float inside_log) {
    return b - c * inside_log;
}

__global__ void fire_neurons_kernel(float *a, float *b, float c, float tau, bool *fired,
				    float *spike_times, float current_time,
				    float next_pre_time) {
    unsigned int neuron_idx = threadIdx.x;
    float x, inside_log, spike_time;
    bool valid_spike;

    a += neuron_idx;
    b += neuron_idx;
    spike_times += neuron_idx;
    x = compute_inside_x(*a, *b, c);
    if (x < 0) {
	*spike_times = INFINITY;
	return;
    }
    x = sqrt(x);
    inside_log = compute_inside_log(*a, *b, x);
    if (inside_log <= 0) {
	*spike_times = INFINITY;
	return;
    }
    spike_time = compute_spike_time(inside_log, tau);
    valid_spike = *spike_times < spike_time && spike_time < next_pre_time;
    if (!valid_spike) {
	*spike_times = INFINITY;
	return;
    }
    *spike_times = spike_time; // Add current_time as spike time is relative to last pre-spike
    *b = apply_reset(*b, c, inside_log);
    *fired = true;
}

static bool fire_neurons(fc_layer_t *layer, float current_time, float next_pre_time) {
    float *a = layer->a;
    float *b = layer->b;
    float c = layer->params.c;
    bool *fired = layer->fired;
    float *spike_times = layer->spike_times;
    unsigned int *n_spikes = layer->n_spikes;
    float tau = layer->params.tau;
    unsigned int n_neurons = layer->params.n_neurons;
    
    cudaMemset(layer->spike_times, 0, n_neurons * sizeof(float));
    do {
        *fired = 0;
	fire_neurons_kernel<<<1, n_neurons>>>(a, b, c, tau, fired, spike_times, current_time,
					      next_pre_time);
	cudaDeviceSynchronize();
	if (!(*fired))
	    break;
	for (int i = 0; i < n_neurons; i++) {
	    if (spike_times[i] == INFINITY)
		continue;
	    layer->post_spikes = spike_list_add(layer->post_spikes, i, current_time + spike_times[i]);
	    if (layer->post_spikes == 0)
		return false;
	    n_spikes[i]++;
	    layer->total_n_spikes++;
	}
    } while (*fired);
    return true;
}

static bool integrate_pre_spike(fc_layer_t *layer, const spike_list_t *pre_spike,
				float next_pre_time) {
    
    update_state(layer, pre_spike);
    if (!fire_neurons(layer, pre_spike->time, next_pre_time))
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

#ifdef __cplusplus
}
#endif
