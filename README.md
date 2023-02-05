# EvSpikeSim (Experimental)

## Project Description
EvSpikeSim is an experimental event-based Spiking Neural Networks (SNNs) simulator written in C++ for high performance and interfaced with Python.
This project aims to provide fast and accurate simulations of sparse SNNs for the development of neuromorphic training algorithms.

## Implemented Features

- Fully-connected layers of Leaky Integrate-and-Fire (LIF) neurons
- Simple Python3 interface compatible with numpy arrays
- Multi-theading on CPU
- NVIDIA GPU support

## Coming Features

- Local gradients
- Spike Time-Dependent Plasticity (STDP)
- Error-Driven Learning
- Direct Feedback Alignement (DFA)
- Convolutional Spiking Layers

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

In this simulator, **membrane time constants are constrained to 2x the synaptic time constants**, i.e. $\tau = 2 \tau_s$. This allows us to isolate a closed-form solution for the spike time and achieve fast event-based inference without the use of numerical solvers. See ref [1, 2] for more details.

## Getting started

Two implementations are available: one for CPUs and one for NVIDIA GPUs.

### CPU Install

#### Dependencies

- g++ >= 7
- cmake >= 3.14

#### Build and install C++ library

To compile the C++ library, run:
```console
mkdir build
cd build
cmake ../core -DNO_TEST=ON
make
```

To install the library:
```
sudo make install
```

#### Build and install Python3 API

Compilation of the C++ library is required to build the Python3 API. 
(See previous section).
Then, run the following commands to build and install the Python3 API:
```
cd python_api
pip3 install -r requirements.txt
python3 setup.py install
```

### GPU Install

#### Dependencies

- g++ >= 7
- cmake >= 3.14
- CUDA Toolkit >= 11

#### Build and install C++ library

To compile the C++ library for NVIDIA GPUs, run:
```console
mkdir build
cd build
cmake ../core -DNO_TEST=ON -DBUILD_GPU=ON
make
```
In some cases, the compute capability of the target GPU (see [this link](https://developer.nvidia.com/cuda-gpus)) 
might need to be provided to cmake:
```
cmake ../core/ -DBUILD_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=XX
```
Simply replace XX by the corresponding compute capability.

Finally, to install the library:
```
sudo make install
```

#### Build and install Python3 API

Compilation of the C++ library is required to build the Python3 API. 
(See previous section).
Then, run the following commands to build and install the Python3 API:
```
cd python_api
pip3 install -r requirements.txt
python3 setup.py install --gpu
```
If the nvcc compiler is not found, try specifying the path to the Cuda home directory in your environment:
```
CUDAHOME=/path/to/cuda/ python3 setup.py install
```

## Usage

### C++ Library

#### Compilation for CPU

For CPU, the pthread library needs to be linked to your project:
```
g++ foo.cpp -std=c++17 -levspikesim -lpthread
```
Compilation requires c++17.

#### Compilation for GPU

For GPU, we recommend to compile all source files using nvcc and specify g++ as host compiler:
```
nvcc foo.cpp -ccbin g++ -std=c++17 -levspikesim
```
Compilation requires c++17.

#### Example

Here is a C++ example of a (very) small fully-connected SNN in EvSpikeSim:
```cpp
#include <evspikesim/Layers/FCLayer.h>
#include <evspikesim/SpikingNetwork.h>

namespace sim = EvSpikeSim;

int main() {
    // Create network
    auto network = sim::SpikingNetwork();

    // Layer parameters
    unsigned int n_inputs = 2;
    unsigned int n_neurons = 3;
    float tau_s = 0.020;
    float threshold = tau_s * 0.2;

    // Add fully-connected layer to the network
    auto desc = sim::FCLayerDescriptor(n_inputs, n_neurons, tau_s, threshold);
    std::shared_ptr<sim::FCLayer> layer = network.add_layer(desc);

    // Set weights
    std::vector<float> weights = {1.0, 0.3,
                                  -0.1, 0.8,
                                  0.5, 0.4};
    std::copy(weights.data(), weights.data() + weights.size(), layer->get_weights().c_ptr());

    // Mutate weight
    layer->get_weights().get(0, 1) -= 0.1;

    // Create input spikes
    std::vector<unsigned int> input_indices = {0, 1, 1};
    std::vector<float> input_times = {1.0, 1.5, 1.2};
    auto input_spikes = sim::SpikeArray(input_indices, input_times);
    input_spikes.sort();

    // Inference
    auto output_spikes = network.infer(input_spikes);

    std::cout << "Input spikes:" << std::endl;
    std::cout << input_spikes << std::endl;

    std::cout << "Output spikes:" << std::endl;
    std::cout << output_spikes << std::endl;

    std::cout << "Output spike counts:" << std::endl;
    for (auto it : layer->get_n_spikes())
        std::cout << it << " ";
    std::cout << std::endl;
    return 0;
}
```
This code can compile with either the CPU or GPU versions of EvSpikeSim.

### Python API

Here is a Python3 example of a (very) small fully-connected SNN in EvSpikeSim:
```python
import evspikesim as sim
import numpy as np


def main():
    # Create network
    network = sim.SpikingNetwork()

    # Layer parameters
    n_inputs = 2
    n_neurons = 3
    tau_s = 0.020
    threshold = tau_s * 0.2

    # Add fully-connected layer to the network
    desc = sim.layers.FCLayerDescriptor(n_inputs, n_neurons, tau_s, threshold)
    layer = network.add_layer(desc)

    # Set weights
    layer.weights = np.array([[1.0, 0.3],
                              [-0.1, 0.8],
                              [0.5, 0.4]], dtype=np.float32) # Weights need to be of type float32

    # Mutate weight
    layer.weights[0, 1] -= 0.1

    # Create input spikes
    input_indices = np.array([0, 1, 1], dtype=np.uint32) # Spike indices need to be of type uint32 or int32
    input_times = np.array([1.0, 1.5, 1.2], dtype=np.float32) # Spike times need to be of type float32
    input_spikes = sim.SpikeArray(input_indices, input_times)
    input_spikes.sort() # Input spikes need to be sorted in time prior to inference

    # Inference
    output_spikes = network.infer(input_spikes)

    print("Input spikes:")
    print(input_spikes)

    print("Output spikes:")
    print(output_spikes)

    print("Output spike counts:")
    print(layer.n_spikes)

if __name__ == "__main__":
    main()
```
This code can run with either the CPU or GPU versions of EvSpikeSim.

## Docker

To avoids the troubles that can be encountered when building EvSpikeSim, we provide Dockerfiles 
for CPU and GPU implementations.

### Build EvSpikeSim CPU Image

From the project root:
```
docker build -t evspikesim_cpu -f docker/cpu/Dockerfile .
```

### Build EvSpikeSim GPU Image

[nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) 
is required to be able to use GPUs in Docker containers.
After installing nvidia-docker, run from the project root:
```
sudo nvidia-docker build -t evspikesim_gpu -f docker/gpu/Dockerfile .
```

## Unit Tests

### C++ Library

After building and/or installing the library, run in the build folder:
```
make tests
./tests/tests
```

### Python3 API

After installing the EvSpikeSim Python package on your system, run the following command:
```
python3 -m unittest discover -p "*_test.py"
```

## References

[1] J. Göltz, L. Kriener, A. Baumbach, S. Billaudelle, O. Breitwieser, B. Cramer, D. Dold, A. F. Kungl, W. Senn, J. Schemmel, K. Meier, & M. A. Petrovici (2021). Fast and energy-efficient neuromorphic deep learning with first-spike times. <em>Nature Machine Intelligence, 3(9), 823–835.</em> <br>
[2] Bacho, F., & Chu, D.. (2022). Exact Error Backpropagation Through Spikes for Precise Training of Spiking Neural Networks. https://arxiv.org/abs/2212.09500 <br>
