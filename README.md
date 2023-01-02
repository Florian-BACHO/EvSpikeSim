# EvSpikeSim (Beta)

## Project Description
EvSpikeSim is an event-based Spiking Neural Networks (SNNs) simulator written in C for high performance and interfaced with Python for easy usage.
This project aims to provide fast and accurate simulations of SNNs for the development of neuromorphic training algorithms.

## Implemented Features

- Python interface with numpy arrays
- Fully-connected layers of Leaky Integrate-and-Fire (LIF) neurons
- GPU support

## Neuron Model

The neuron model implemented in this simulator is the Current-Based Leaky Integrate-and-Fire (CuBa LIF) neuron. Its defined by the following system of ordinary differential equations:
```math
\left\{\begin{matrix}
\frac{du_i(t))}{dt}=-\frac{u_i(t))}{\tau} + g_i(t) - \underbrace{\vartheta \delta\left ( u(t) - \vartheta \right )}_{\text{Reset}}\\
\frac{dg_i(t))}{dt}=-\frac{g_i(t))}{\tau_s} + \underbrace{\sum_{j} w_{i,j} \sum_{t_j} \delta\left ( t_j - t \right)}_{\text{Pre-synaptic spikes}}
\end{matrix}\right.
```
where $g_i(t)$ and $u_i(t)$ are the synaptic current and membrane potential of the post-synaptic neuron $i$, $\tau_s$ and $\tau$ are the synaptic and membrane time constants, $w_{i,j}$ is the weight between the post-synaptic neuron $i$ and the pre-synaptic neuron $j$, $t_j$ is a pre-synaptic spike timing, $\vartheta$ is the threshold of the neuron and $\delta(x)$ is the Dirac Delta function.

When the membrane potential reaches its threshold, i.e. $u_i(t)=\vartheta$, a post-synaptic spike is emitted by the neuron $i$ at time $t$.

## Build and Install

### Python3

Requirements:
- cmake (>= 3.16)
- Python3 (>= 3.8)
- Numpy (>=1.20)
- Cuda (if install for GPU) (>=11.6)

The following command builds the EvSpikeSim Python3 package:
```console
python setup.py build
```
To install the build and install the package, run:
```console
python setup.py install
```
For GPU builds and/or install, first specify the path to the Cuda directory in the `CUDAHOME` add the `--gpu` argument to the command:
```console
export CUDAHOME=/PATH/TO/cuda
python setup.py install --gpu
```
If installed for GPU, the default GPU device will be used instead of the CPU during inference.

If the installation fails due to permission issues, try installing in the user site-package using the `--user` argument:
```console
python setup.py install --user
```

### C Language

#### Requirements

- cmake (>= 3.16)
- Cuda (if install for GPU) (>=11.6)

Copy either the `CMakeLists_cpu.txt` or the `CMakeLists_gpu.txt` under the name 'CMakeLists.txt' to compile for CPU or GPU.
Then, run the following sequence of commands to build the static library:
```console
mkdir build
cd build
cmake ..
make
```
This outputs the compiled library under the name `libevspikesim.a`

To install the library and its header files, run the command:
```console
sudo make install
```

## Usage

### Python3

Import the simulator as follow:
```python
import evspikesim as sim

...
```

#### Create a network

A SNN is created by instanciating a `Network` object and sequentially adding layers:
```python
...

n_inputs = 100
n_neurons = 1024
tau_s = 0.010 # 10 ms
threshold = 0.5 * tau_s # Relative to tau_s as the amplitude of the PSP kernel is proportional to tau_s

network = sim.Network() # Creates empty network
network.add_fc_layer(n_inputs, n_neurons, tau_s, threshold) # Add a fully-connected layer of LIF neurons

...
```
The `add_fc_layer` method adds a fully-connected layer of LIF neurons to the network. 
It takes four arguments:
- the number of inputs of the layer (unsigned integer)
- the number of neurons in the layer (unsigned integer)
- the synaptic time constant $\tau_s$ in seconds (floating point)
- the threshold $\vartheta$ of the neurons (floating point)

Note that, as the PSP kernel of the implemented neuron model depends on the synaptic time constant $\tau_s$, it is good practice to scale the threshold by $\tau_s$ and keep a constant factor.

#### Inference

To infer the network, two 1-D numpy arrays need to be created to store the input spike indices (i.e. input neurons) and input spike timings.
These arrays are then given as argument to the infer method:
```python
import numpy as np
...

# Define input spikes
input_spike_indices = np.array([21, 84, 42], dtype=np.uint32)
input_spike_times = np.array([0.013, 0.009, 0.012], dtype=np.float32)

# Infer the network
network.reset()
network.infer(input_spike_indices, input_spike_times)

# Get output spikes and counts
output_spike_indices, output_spike_times = network.output_layer.spikes
output_spike_counts = network.output_layer.spike_counts

...
```
Spike indices and spike times arrays must have the same shape but do not have to be sorted in time.  Note that input spike indices MUST be of type `np.uint32` and input spike timings MUST be of type `np.float32`. The network also has to be reset before inference. Finally, the output layer can be accessed through the `output_layer` member of the network object to get the output results.

#### Weights

By default, weights of a layer are randomly initialized with a random uniform distribution between -1 and 1.

Weights of a specific layer can be accessed and mutated as follows:
```python
...

layer_idx = 0
weights = network[layer_idx].weights

weights[42, 21] += 0.42

...
```
`network[layer_idx]` returns the corresponding layer at the given index `layer_idx`. The `weights` attribute stores the weights in a 2-D numpy array of shape `(n_neurons, n_inputs)`. Mutating this array change the weights of the network. This can be used to apply changes of weights during training.

#### Example

The following example shows the inference of three 2-input LIF neuron with weights `[[1.0, 2.0], [-0.1, 0.8], [0.5, 0.4]]` and driven by two input spikes `(idx: 0, time: 0.1)` and `(idx: 1, time: 0.15)`
```python
import numpy as	np
import evspikesim as sim

# Network definition                                                                                
network = sim.Network()
network.add_fc_layer(2, 3, 0.020, 0.020 * 0.1)
network[0].weights = np.array([[1.0, 2.0],
                               [-0.1, 0.8],
                               [0.5, 0.4]], dtype=np.float32)

# Input spikes                                                                                      
input_spike_indices = np.array([0, 1], dtype=np.uint32)
input_spike_times = np.array([0.013, 0.009], dtype=np.float32)

network.reset()
network.infer(input_spike_indices, input_spike_times)

# Get output spikes and counts                                                                      
output_spike_indices, output_spike_times = network.output_layer.spikes
output_spike_counts = network.output_layer.spike_counts

print(output_spike_indices)
print(output_spike_times)
print(output_spike_counts)
```

### C Language

#### Linkage

To compile your C projects with our simulator, the `evspikesim` and the math libraries need to be linked:
```console
gcc -o foo foo.c -levspikesim -lm
```
or
```console
ld -o foo foo.o -levspikesim -lm
```
For GPU, compile as follows:
```console
gcc -o foo foo.c -levspikesim -lm -lcuda -lcudart -lstdc++
```

#### Spike List

Spike events are stored in a doubly-circular linked list where nodes are of type `spike_list_t`. This structure has the following definition:
```c
typedef struct spike_list {
    struct spike_list *prev;
    struct spike_list *next;
    unsigned int index;
    float time;
} spike_list_t;
```
Each node of this linked list represent a spike with a given neuron index and spike timing. Nodes are sorted in ascending order of time.
An empty list is represented by a `NULL` pointer. New spikes can be added to a list using the `spike_list_add` function, such as:
```c
#include <assert.h>
#include <evspikesim/spike_list.h>

...

unsigned int spike_idx = 42;
float spike_time = 0.013;
spike_list_t *inputs = 0;

inputs = spike_list_add(inputs, spike_idx, spike_time);
assert(inputs != 0);

... // Do something

spike_list_destroy(inputs);

...
```
The function returns `NULL` if dynamic allocation failed; Do not forget to free the allocated memory by calling the `spike_list_destroy` function.

To iterate through a spike list, first check if the spike list is empty using the `spike_list_empty` function than loop through the nodes until the `next` attribute is equals to the first node:
```c
...

spike_list_t *current_node = start_node;

if (spike_list_empty(start_node))
  return; // Empty list
  
do {
  ... // Do something
  current_node = current_node->next;
} while (current_node != start_node);

...
```

#### Create a network

A network is created by instanciating a `network_t` structure and initializing it with the `network_init()` function. Fully-connected layers are created by calling the `network_add_fc_layer` function with a `fc_layer_params_t` structure and an initialization function with prototype `float init_fct(void)` (can be set to `NULL` for no weight initialization but will require manual weights set):
```c
#include <assert.h>
#include <evspikesim/network.h>
#include <evspikesim/fc_layer_params.h>
#include <evspikesim/random.h>

...

// Initialization function with uniform distribution between -1 and 1
static float init_fct() {
  return random_uniform_float(-1.0f, 1.0f);
}

...


unsigned int n_inputs = 100
unsigned int n_neurons = 1024
float tau_s = 0.010 # 10 ms
float threshold = 0.5 * tau_s // Relative to tau_s as the amplitude of the PSP kernel is proportional to tau_s
fc_layer_params_t layer_params = fc_layer_params_new(n_inputs, n_neurons, tau_s, threshold);

network_t network = network_init();
const fc_layer_t *layer = network_add_fc_layer(&network, layer_params, &init_fct);

assert(layer != 0);

... // Inference

network_destroy(&network);

...
```
The `network_add_fc_layer` function adds a fully-connected layer of LIF neurons to the given network. 
It takes four arguments:
- the number of inputs of the layer (unsigned int)
- the number of neurons in the layer (unsigned int)
- the synaptic time constant $\tau_s$ in seconds (float)
- the threshold $\vartheta$ of the neurons (float)

Do not forget to free the allocated memory by calling the `network_destroy` function when the network needs to be deleted.

Note that, as the PSP kernel of the implemented neuron model depends on the synaptic time constant $\tau_s$, it is good practice to scale the threshold by $\tau_s$ and keep a constant factor.

#### Inference

To infer the network, an input spike list must be created and sent to the `network_infer` method:
```c
...

spike_list_t *input_spikes;
spike_list_t *output_spikes;
network_t network = network_init();

... // Create input spikes and network

network_reset(&network);
output_spikes = network_infer(&network, input_spikes);

spike_list_print(output_spikes); // Print output spikes

...

spike_list_destroy(input_spikes);
network_destroy(&network);
```

#### Layer

Layers can also be accessed through the `network_t` structure as follows:
```c
#include <evspikesim/fc_layer.h>
#include <evspikesim/layer_type.h>

...

network_t *network = network_init();

... // Create network

unsigned int layer_idx = 1;
fc_layer *layer;

assert(layer_idx < network.n_layers);
assert(network.layer_types[layer_idx] == FC); // Check layer type before cast

layer = (fc_layer_t *)network.layers[layer_idx];

```
The `layer_type` attribute stores the type of the corresponding layers in the `layers` attribute. It is always good practice to test the layer type before casting to prevent from invalid memory access. 

The `fc_layer_t` structure is defined as follows:
```c
typedef struct {
    fc_layer_params_t params; // Parameter structure sent at creation
    float *weights; // Neurons weights of size (n_neurons * n_inputs)
    float *a; // Internal state used during inference. Do not mutate.
    float *b; // Internal state used during inference. Do not mutate.
    spike_list_t *post_spikes; // Post synaptic spikes of the layer. Populated during inference
    unsigned int *n_spikes; // Number of post-synaptic spikes fired by each neuron. Populated during inference
    unsigned int total_n_spikes; // Total number of post-synaptic spikes. Set during inference
} fc_layer_t;
```
Parameters and weights can be freely mutated. To change a specific weight of a neuron at index `neuron_idx` and synapse `input_idx`, use the following indexing:
```c
layer->weight[neuron_idx * layer->params.n_inputs + input_idx] += 0.2f;
```
The `n_spikes` function has a size of (n_neurons,) and stores the neurons spike counts during inference. Spike events can be accessed through the `post_spikes` attribute.

#### Example

The following example shows the inference of three 2-input LIF neuron with weights `[[1.0, 2.0], [-0.1, 0.8], [0.5, 0.4]]` and driven by two input spikes `(idx: 0, time: 0.1)` and `(idx: 1, time: 0.15)`
```c
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
    fc_layer_params_t params = fc_layer_params_new(2, 3, 0.020f, 0.020 * 0.1);
    network_t network = network_init();
    fc_layer_t *layer = network_add_fc_layer(&network, params, 0);
    const spike_list_t *output_spikes;

    if (layer == 0)
        return 1;
    
    // Copy weights
    fc_layer_set_weights(layer, weights);

    // Inference                                                                                     
    network_reset(&network);
    output_spikes = network_infer(&network, input_spikes);
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
    network_destroy(&network);
    
    return 0;
}

```

## Coming Features

- Mini-batch learning
- Local gradients computation
- Spike Time-Dependent Plasticity (STDP)
- Direct Feedback Alignement (DFA)
- Convolutional Spiking Layers
- Recurrence
