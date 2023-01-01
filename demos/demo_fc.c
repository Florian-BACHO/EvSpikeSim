#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "spike_list.h"
#include "fc_layer.h"
#include "fc_layer_params.h"
#include "network.h"
#include "random.h"

static inline float init_fct(void) {
    return random_uniform_float(-1.0f, 1.0f);
}

spike_list_t *generate_random_inputs(unsigned int n_spikes, unsigned int n_neurons,
				     float max_time) {
    spike_list_t *inputs = 0;
    unsigned int spike_idx;
    float spike_time;

    for (unsigned int i = 0; i < n_spikes; i++) {
	spike_idx = random_uniform_int(0, n_neurons);
	spike_time = random_uniform_float(0.0f, max_time);
	inputs = spike_list_add(inputs, spike_idx, spike_time);
	if (inputs == 0)
	    return 0;
    }
    return inputs;
}

void print_spike_counts(unsigned int *n_spikes, unsigned int n_neurons) {
    for (unsigned int i = 0; i < n_neurons; i++)
	printf("%d ", n_spikes[i]);
    printf("\n");
}

int main() {
    random_set_seed(42); // Set seed for reproducibility
    // Inputs
    unsigned int n_inputs = 100;
    unsigned int n_input_spikes = 30;
    float max_input_spike_time = 0.010f; // 10 ms
    spike_list_t *input_spikes = generate_random_inputs(n_input_spikes, n_inputs,
							max_input_spike_time);
    
    if (input_spikes == 0)
	return 1;

    // Network definition
    unsigned int n_neurons_hidden_1 = 1024;
    unsigned int n_neurons_hidden_2 = 512;
    unsigned int n_neurons_out = 10;
    float tau_s = 0.010f; // Synaptic time constant of 10 ms. Membrane time constant is tau = 2 * tau_s = 20 ms
    float threshold_hidden_1 = 0.5f * tau_s;
    float threshold_hidden_2 = 8.0f * tau_s;
    float threshold_out = 2 * 4.0f * tau_s;
    // Create parameter structures for layers
    fc_layer_params_t params_hidden_1 = fc_layer_params_new(n_inputs, n_neurons_hidden_1,
							    tau_s, threshold_hidden_1);
    fc_layer_params_t params_hidden_2 = fc_layer_params_new(n_neurons_hidden_1, n_neurons_hidden_2,
							    tau_s, threshold_hidden_2);
    fc_layer_params_t params_out = fc_layer_params_new(n_neurons_hidden_2, n_neurons_out,
						       tau_s, threshold_out);
    const spike_list_t *output_spikes;

    // Network instanciation
    network_t network = network_init();
    const fc_layer_t *hidden_layer_1 = network_add_fc_layer(&network, params_hidden_1, &init_fct);
    const fc_layer_t *hidden_layer_2 = network_add_fc_layer(&network, params_hidden_2, &init_fct);
    const fc_layer_t *output_layer = network_add_fc_layer(&network, params_out, &init_fct);

    // Create layers
    if (hidden_layer_1 == 0 || hidden_layer_2 == 0 || output_layer == 0)
	return 1;

    printf("Input spikes:\n");
    spike_list_print(input_spikes);

    // Inference
    output_spikes = network_infer(&network, input_spikes);
    if (output_spikes == 0)
	return 1;

    printf("\nOutput spikes:\n");
    spike_list_print(output_spikes);
    printf("\nHidden layer 1 spike counts:\n");
    print_spike_counts(hidden_layer_1->n_spikes, hidden_layer_1->params.n_neurons);
    printf("\nHidden layer 2 counts:\n");
    print_spike_counts(hidden_layer_2->n_spikes, hidden_layer_2->params.n_neurons);
    printf("\nOutput spike counts:\n");
    print_spike_counts(output_layer->n_spikes, output_layer->params.n_neurons);

    // Free memory
    spike_list_destroy(input_spikes);
    network_destroy(&network);
}
