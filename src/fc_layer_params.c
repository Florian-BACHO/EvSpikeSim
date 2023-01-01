#include "fc_layer_params.h"

fc_layer_params_t fc_layer_params_new(unsigned int n_inputs, unsigned int n_neurons, float tau_s,
				      float threshold) {
    fc_layer_params_t out;

    out.n_inputs = n_inputs;
    out.n_neurons = n_neurons;
    out.tau_s = tau_s;
    out.tau = 2 * tau_s;
    out.threshold = threshold;
    out.c = threshold / out.tau;
    return out;
}
