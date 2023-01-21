#ifndef FC_LAYER
#define FC_LAYER

#include <pthread.h>
#include "spike_list.h"
#include "fc_layer_params.h"
#include "infer_thread_pool.h"

#define BUFFER_SIZE 64

typedef struct {
    fc_layer_params_t params;
    float *weights;
    spike_list_t *post_spikes;
    unsigned int *n_spikes;
    unsigned int total_n_spikes;
    // For internal use
    float *a;
    float *b;
    infer_thread_pool_t *thread_pool; // used on CPU only; weak pointer: owned by network_t
    pthread_mutex_t layer_mutex; // used on CPU only to lock the layer in threading setting
    bool *buffer_full; // used with GPU only
    float *spike_time_buffer; // used with GPU only
} fc_layer_t;

fc_layer_t *fc_layer_new(fc_layer_params_t params, float (*init_fct)(void),
			 infer_thread_pool_t *thread_pool);
void fc_layer_destroy(fc_layer_t *layer);
void fc_layer_reset(fc_layer_t *layer);
void fc_layer_set_weights(fc_layer_t *layer, const float *new_weights);
const spike_list_t *fc_layer_infer(fc_layer_t *layer, const spike_list_t *pre_spikes,
				   unsigned int total_n_spikes);

#endif
