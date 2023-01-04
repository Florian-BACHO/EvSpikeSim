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
    fc_layer_t *out;
    
    if (cudaMallocManaged(&out, sizeof(fc_layer_t)) ||
	cudaMallocManaged(&out->weights, params.n_neurons * params.n_inputs * sizeof(float)) ||
	cudaMallocManaged(&out->n_spikes, params.n_neurons * sizeof(unsigned int)) ||
	cudaMallocManaged(&out->a, params.n_neurons * sizeof(float)) ||
	cudaMallocManaged(&out->b, params.n_neurons * sizeof(float)) ||
	cudaMallocManaged(&out->buffer_full, sizeof(bool)) ||
	cudaMallocManaged(&out->spike_time_buffer, params.n_neurons * BUFFER_SIZE * sizeof(float)))
	return 0;
    out->params = params;
    cudaMemset(out->a, 0, params.n_neurons * sizeof(float));
    cudaMemset(out->b, 0, params.n_neurons * sizeof(float));
    cudaMemset(out->n_spikes, 0, params.n_neurons * sizeof(unsigned int));
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
    cudaFree(layer->buffer_full);
    cudaFree(layer->spike_time_buffer);
    cudaFree(layer->n_spikes);
    spike_list_destroy(layer->post_spikes);
    cudaFree(layer);
}

void fc_layer_reset(fc_layer_t *layer) {
    cudaMemset(layer->a, 0, layer->params.n_neurons * sizeof(float));
    cudaMemset(layer->b, 0, layer->params.n_neurons * sizeof(float));
    cudaMemset(layer->n_spikes, 0, layer->params.n_neurons * sizeof(unsigned int));
    layer->total_n_spikes = 0;
    spike_list_destroy(layer->post_spikes);
    layer->post_spikes = 0;
}

void fc_layer_set_weights(fc_layer_t *layer, const float *new_weights) {
    size_t size = layer->params.n_inputs * layer->params.n_neurons * sizeof(float);

    cudaMemcpy(layer->weights, new_weights, size, cudaMemcpyDefault);
}

static bool convert_input_spike_list(const spike_list_t *spikes, unsigned int n_spikes,
				     unsigned int **indices, float **times) {
    if (cudaMallocManaged(indices, (n_spikes + 1) * sizeof(unsigned int)) ||
	cudaMallocManaged(times, (n_spikes + 1) * sizeof(float)))
	return false;
    for (unsigned int i = 0; i < n_spikes; i++) {
	(*indices)[i] = spikes->index;
	(*times)[i] = spikes->time;
	spikes = spikes->next;
    }
    (*indices)[n_spikes] = 0;
    (*times)[n_spikes] = INFINITY;
    return true;
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

__global__ void init_spike_time_buffer(float *buffer) {
    unsigned int neuron_idx = threadIdx.x;
    unsigned int spike_idx = blockIdx.x;

    buffer[neuron_idx * BUFFER_SIZE + spike_idx] = INFINITY;
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

__device__ void fire(fc_layer_t *layer, unsigned int neuron_idx, float current_time,
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
				 *n_spike_buffer] = current_time + spike_time; // Add current_time as spike time is relative to last pre-spike
	*b = apply_reset(*b, c, inside_log);
	(*n_spike_buffer)++;
    }
    *(layer->buffer_full) = true;
}

__inline__ __device__ float get_next_time(const float *times) {
    return (times[0] == 0 || times[1] == 0) ? (INFINITY) : (times[1] - times[0]);
}

__device__ void apply_decay_and_weight(fc_layer_t *layer, unsigned int neuron_idx, float delta_t,
				       unsigned int pre_idx) {
    float weight = layer->weights[neuron_idx * layer->params.n_inputs + pre_idx];
    float exp_tau = exp(-delta_t / layer->params.tau);
    float exp_tau_s = exp_tau * exp_tau; // Because tau = 2 * tau_s, squaring exp_tau gives exp_tau_s

    layer->a[neuron_idx] = layer->a[neuron_idx] * exp_tau_s + weight;
    layer->b[neuron_idx] = layer->b[neuron_idx] * exp_tau + weight;
}

__global__ void infer_kernel(fc_layer_t *layer, const unsigned int *indices, const float *times) {
    unsigned int neuron_idx = threadIdx.x;
    unsigned int n_spike_buffer = 0;
    float next_time = 0.0;

    while (*times != INFINITY && n_spike_buffer < BUFFER_SIZE) {
	apply_decay_and_weight(layer, neuron_idx, next_time, *indices);
	next_time = get_next_time(times);
	fire(layer, neuron_idx, *times, next_time, &n_spike_buffer);
	indices++;
	times++;
    }
}

const spike_list_t *fc_layer_infer(fc_layer_t *layer, const spike_list_t *pre_spikes,
				   unsigned int n_pre_spikes) {
    unsigned int *input_indices;
    float *input_times;

    if (spike_list_empty(pre_spikes) || !convert_input_spike_list(pre_spikes, n_pre_spikes,
								  &input_indices, &input_times))
        return 0;
    //do {
    *(layer->buffer_full) = false;
    init_spike_time_buffer<<<BUFFER_SIZE, layer->params.n_neurons>>>(layer->spike_time_buffer);
    cudaDeviceSynchronize();
    infer_kernel<<<1, layer->params.n_neurons>>>(layer, input_indices, input_times);
    cudaDeviceSynchronize();
    if (!process_buffer_spikes(layer))
	return 0;
    //} while(*(layer->buffer_full));
    cudaFree(input_indices);
    cudaFree(input_times);
    return layer->post_spikes;
}

#ifdef __cplusplus
}
#endif
