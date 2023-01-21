#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <evspikesim/spike_list.h>
#include <evspikesim/fc_layer.h>
#include <evspikesim/fc_layer_params.h>
#include <evspikesim/network.h>

int main(void) {
    // Inputs spikes                                                                                  
    float max_input_spike_time = 0.010f; // 10 ms                                                    
    spike_list_t *input_spikes = 0;
    const float weights[] = {1.0, 2.0,
                             -0.1, 0.8,
                             0.5, 0.4};

    input_spikes = spike_list_add(input_spikes, 0, 0.013f);
    if (input_spikes == 0)
        return 1;
    input_spikes = spike_list_add(input_spikes, 1, 0.009f);
    if (input_spikes == 0)
        return 1;

    // Network definition
    unsigned int n_threads = 0;
    fc_layer_params_t params = fc_layer_params_new(2, 3, 0.020f, 0.020 * 0.1);
    network_t *network = network_new(n_threads);

    if (network == 0)
	return 1;
    
    fc_layer_t *layer = network_add_fc_layer(network, params, 0);
    const spike_list_t *output_spikes;

    if (layer == 0)
        return 1;
    
    // Copy weights
    fc_layer_set_weights(layer, weights);

    // Inference
    
    network_reset(network);
    output_spikes = network_infer(network, input_spikes, 2);
    if (output_spikes == 0)
        return 1;

    // Print spikes                                                                          
    spike_list_print(output_spikes);
    
    // Print spike count                                                                             
    for (unsigned int i = 0; i < layer->params.n_neurons; i++)
        printf("%d ", layer->n_spikes[i]);
    printf("\n");

    // Free memory
    spike_list_destroy(input_spikes);
    network_destroy(network);

    return 0;
}
