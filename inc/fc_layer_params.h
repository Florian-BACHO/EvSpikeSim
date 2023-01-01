#ifndef FC_LAYER_PARAMS
#define FC_LAYER_PARAMS

typedef struct fc_layer_params {
    unsigned int n_inputs;
    unsigned int n_neurons;
    float tau_s;
    float tau;
    float threshold;
    float c;
} fc_layer_params_t;

fc_layer_params_t fc_layer_params_new(unsigned int n_inputs, unsigned int n_neurons, float tau_s,
				      float threshold);

#endif
