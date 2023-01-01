#ifndef FC_LAYER
#define FC_LAYER

#include "spike_list.h"
#include "fc_layer_params.h"

typedef struct {
    fc_layer_params_t params;
    float *weights;
    float *a;
    float *b;
    spike_list_t *post_spikes;
    unsigned int *n_spikes;
    unsigned int total_n_spikes;
} fc_layer_t;

fc_layer_t *fc_layer_new(fc_layer_params_t params, float (*init_fct)());
void fc_layer_destroy(fc_layer_t *layer);
void fc_layer_reset(fc_layer_t *layer);
void fc_layer_set_weights(fc_layer_t *layer, const float *new_weights);
const spike_list_t *fc_layer_infer(fc_layer_t *layer, const spike_list_t *pre_spikes);

#endif
