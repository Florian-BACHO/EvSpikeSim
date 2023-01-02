# EvSpikeSim (Beta)

## Project Description
EvSpikeSim is an event-based Spiking Neural Networks (SNNs) simulator written in C for high performance and interfaced with Python for easy usage.
This project aims to provide fast and accurate simulations of SNNs for the development of neuromorphic training algorithms.

## Implemented Features

- Python interface with numpy arrays
- Fully-connected layers of Leaky Integrate-and-Fire (LIF) neurons

## Requirements

- cmake (>= 3.16)
- Python3 (>= 3.8)
- Numpy (>=1.20)

## Build and Install

### Python3

The following command builds the EvSpikeSim Python3 package:
```console
python setup.py build
```
To install the build and install the package, run:
```console
python setup.py install
```
If the installation fails due to permission issues, try installing in the user site-package using the `--user` argument:
```console
python setup.py install --user
```

### C Language

Run the following sequence of commands to build the C static library:
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

#### Spike List

#### Create a network

#### Layer

#### Inference

#### Weights

## Coming Features

- GPU support
- Mini-batch learning
- Local gradients computation
- Spike Time-Dependent Plasticity (STDP)
- Direct Feedback Alignement (DFA)
- Convolutional Spiking Layers
- Recurrence
