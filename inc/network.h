#ifndef NETWORK
#define NETWORK

#include "fc_layer_params.h"
#include "fc_layer.h"
#include "layer_type.h"
#include "spike_list.h"


typedef struct network {
    unsigned int n_layers;
    layer_type_t *layer_types;
    void **layers;
} network_t;

network_t network_init();
fc_layer_t *network_add_fc_layer(network_t *network, fc_layer_params_t params,
				 float (*init_fct)());
void network_destroy(network_t *network);
void network_reset(network_t *network);
const spike_list_t *network_infer(network_t *network, const spike_list_t *input_spikes,
				  unsigned int n_pre_spikes);

#endif
