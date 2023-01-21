#include <stdlib.h>
#include <string.h>
#include "network.h"
#include "fc_layer.h"

inline network_t *network_new(unsigned int n_threads) {
    network_t *out = malloc(sizeof(network_t));

    if (out == 0)
	return 0;
    memset(out, 0, sizeof(network_t));
    if (n_threads == 0)
	return out;
    out->thread_pool = infer_thread_pool_new(n_threads);
    if (out->thread_pool == 0)
	return 0;
    return out;
} 

fc_layer_t *network_add_fc_layer(network_t *network, fc_layer_params_t params,
				 float (*init_fct)(void)) {
    fc_layer_t *new_layer = fc_layer_new(params, init_fct, network->thread_pool);

    network->n_layers++;
    network->layer_types = realloc(network->layer_types, network->n_layers * sizeof(layer_type_t));
    network->layers = realloc(network->layers, network->n_layers * sizeof(void *));
    if (new_layer == 0 || network->layer_types == 0 || network->layers == 0)
	return 0;
    network->layer_types[network->n_layers - 1] = FC;
    network->layers[network->n_layers - 1] = (void *)new_layer;
    return new_layer;
}

void network_destroy(network_t *network) {
    for (unsigned int i = 0; i < network->n_layers; i++) {
	switch (network->layer_types[i]) {
	case FC:
	    fc_layer_destroy((fc_layer_t *)network->layers[i]);
	    break;
	default:
	    break;
	}
    }
    if (network->thread_pool != 0)
	infer_thread_pool_destroy(network->thread_pool);
    free(network->layer_types);
    free(network->layers);
    free(network);
}

void network_reset(network_t *network) {
    for (unsigned int i = 0; i < network->n_layers; i++) {
	switch (network->layer_types[i]) {
	case FC:
	    fc_layer_reset((fc_layer_t *)network->layers[i]);
	    break;
	default:
	    break;
	}
    }
}

const spike_list_t *network_infer(network_t *network, const spike_list_t *spikes,
				  unsigned int n_pre_spikes) {
    for (unsigned int i = 0; i < network->n_layers; i++) {
	switch (network->layer_types[i]) {
	case FC:
	    spikes = fc_layer_infer((fc_layer_t *)network->layers[i], spikes, n_pre_spikes);
	    if (spikes == 0)
		return 0; // Dynamic allocation failed
	    n_pre_spikes = ((fc_layer_t *)network->layers[i])->total_n_spikes;
	    break;
	default:
	    break;
	}
    }
    return spikes;
}
