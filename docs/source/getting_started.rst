===============
Getting Started
===============

This tutorial introduces the basics of EvSpikeSim, from linkinng your project with the library to
creating and running your first Spiking Neural Network (SNN).

.. contents:: Table of Contents

EvSpikeSim C++
==============

In this section, we describe how to link and compile your project with EvSpikeSim and create your first SNN in C++.

Compile for CPU
---------------

If you installed EvSpikeSim for CPU, you can compile and link your project with EvSpikeSim by simply adding the
``-levspikesim`` argument at linkage, such as:

.. code-block:: bash

    g++ foo.cpp -std=c++17 -levspikesim

.. note::
    EvSpikeSim internally uses C++17 features. Therefore, your code also needs to be compiled with C++17.
    This is done by adding the ``-std=c++17`` argument to g++.

Compile for GPUs
----------------

If you installed EvSpikeSim for GPU, we recommend to compile all source files using nvcc and specify g++ as host
compiler. You should then link EvSpikeSim ``-levspikesim`` argument at linkage, such as:

.. code-block:: bash

    nvcc foo.cpp -ccbin g++ -std=c++17 -levspikesim

Includes and Namespace
----------------------

Each class of EvSpikeSim is located in its own file under the ``evspikesim`` directory. Therefore, to include the
`SpikingNetwork` class to a source file, add the following include directive:

.. code-block:: cpp

    #include <evspikesim/SpikingNetwork.h>

Some classes are contained into sub-directories. For example, the ``Layer`` class and its sub-classes are located in
the ``Layers`` subdirectory, leading to the following include directive:

.. code-block:: cpp

    #include <evspikesim/Layers/Layer.h>

All the content provided by EvSpikeSim are contained within the library namespace ``EvSpikeSim`` to avoid any
name conflict. To make your code short, you can redefine the namespace to a shorter version:

.. code-block:: cpp

    namespace sim = EvSpikeSim;

or remove the namespace:

.. code-block:: cpp

    using namespace EvSpikeSim;

In this tutorial, we will use the first method.

Spike
-----

Event-based simulations are driven by events. In EvSpikeSim, those events are the spikes that are received and fired
by neurons. Each spike event is represented by:

- its location;
- the time at which it occurs.

Such a representation of events is similar to the Address-Event Representation protocol (or AER) used for inter-chip
communication between neuromorphic hardware.

In EvSpikeSim, spikes are represented by the ``Spike`` structure defined as:

.. code-block:: cpp

    struct Spike {
        unsigned int index; // Index (in the layer) of the neuron that fired the spike.
        float time; // Timing of the spike.
    }

The ``index`` attribute is the index of the neuron that fired the spike, **in the layer where the event occured**.
For example, ``index=2`` if the third neuron of the layer fired the spike.
The ``time`` attribute is simply the timing of the spike **in seconds**.
Therfore, a spike can be instanciated as follows:

.. code-block:: cpp

    #include <evspikesim/Spike.h>

    ...

    sim::Spike spike(42, 0.021);

This example, instanciates and construct a spike event at the neuron with index 42 and at time 0.021 seconds.

Moreover, spikes are comparable. By comparing spikes, only their timings are taken into account and not their indices.
All the standard comparators are available, i.e. ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``.

.. note::
    For the , ``==`` and , ``!=`` operators, spike timing are compared with a time precision of Spike::epsilon
    (by default: 1e-6, or 1 μs). See C++ API documentation for more details.

Finally, spikes are printable on streams:

.. code-block:: cpp

    std::cout << spike << std::endl;

Spike Arrays
------------

Spike trains are sequences of spikes. In EvSpikeSim, spike trains are represented by an array of ``Spike`` of type
``SpikeArray``. The ``SpikeArray`` effectively stores and sorts in time all the spikes of a layer, facilitating
iterations in chronological order during simulations.

To create a spike array, simply instanciate a new ``SpikeArray`` object as follows:

.. code-block:: cpp

    #include <evspikesim/SpikeArray.h>

    ...

    sim::SpikeArray arr();

Alternatively, spike arrays can be created with a vectors of spikes indices and spike timings, such as:

.. code-block:: cpp

    std::vector<unsigned int> indices = {1, 42, 21};
    std::vector<float> times = {0.030, 0.0017, 0.012};
    sim::SpikeArray arr(indices, times);

which directly populates the array with new ``Spike`` objects.

To add a new spike, call the ``add`` method:

.. code-block:: cpp

    sim::SpikeArray arr();
    sim::Spike spike(42, 0.021);

    arr.add(spike);

The add method can also be called with vectors of spikes indices and spike timings, such as:

.. code-block:: cpp

    sim::SpikeArray arr();
    std::vector<unsigned int> indices = {1, 42, 21};
    std::vector<float> times = {0.030, 0.0017, 0.012};

    arr.add(indices, times);

A spike array can be sorted in time by calling the ``sort`` method:

.. code-block:: cpp

    arr.sort();

Finally, spike arrays are printable on streams:

.. code-block:: cpp

    std::cout << arr << std::endl;

Create a Spiking Network
------------------------

SNNs in EvSpikeSim are feedforward networks. This means that layers are successively simulated and do not
form any cycle or recurrence.

To create a new (empty) SNN, instanciate a ``SpikingNetwork`` object:

.. code-block:: cpp

    #include <evspikesim/SpikingNetwork.h>

    ...

    sim::SpikingNetwork net;

Add Layers of Spiking Neurons
------------------------------

.. note::
    Only a single type of layer (FCLayer) is currently available in EvSpikeSim. New type of layer, such as convolution,
    will be added in future releases.

So far, ``SpikingNetwork`` object is empty and requires layers to do meaniningful computation.
All layers in EvSpikeSim inherit from the ``Layer`` base class.
To add layer a layer to it, call the ``add_layer`` method with a layer class as template and the arguments of the new
layer's constructor as parameters. For example, to add a layer of Fully-Connected (FC) neurons:

.. code-block:: cpp

    #include <evspikesim/SpikingNetwork.h>
    #include <evspikesim/Layers/FCLayer.h>
    #include <evspikesim/Initializers/UniformInitializer.h>

    ...

    sim::SpikingNetwork net; // Network

    unsigned int n_inputs = 100; // Number of inputs.
    unsigned int n_neurons = 1000; // Number of neurons in the layer.
    float tau_s = 0.010; // Synaptic time constant of 10 milliseconds.
    float threshold = 0.1; // Threshold of the neurons.
    sim::RandomGenerator gen; // Random generator for initializer
    sim::UniformInitializer init(gen); // Uniform weight initializer

    std::shared_ptr<sim::FCLayer> layer = net.add_layer<sim::FCLayer>(n_inputs, n_neurons, tau_s, threshold, init); // Add layer

Several parameters are required for the construction of a FCLayer object:

- the number of input neurons, typically the number of neurons in the previous layer;
- the number of neurons in the layer;
- the synaptic time constant controlling the decay of the neurons;
- the threshold of the neurons;
- and a weight initializer.

Here, a new layer of 1000 fully-connected neurons receiving 100 inputs with a synaptic time constant of 10 milliseconds
and a threshold of 0.1 is added to the network. Its weights are initialized with a uniform distribution between -1 and 1
(i.e. the default lower and upper bounds of ``UniformInitializer``). Finally, the ``add_layer`` method returns a
shared pointer on the newly created ``FCLayer`` object.

Access Layers
-------------

Layers in a spiking network can also be accessed as follows:

.. code-block:: cpp

    std::shared_ptr<sim::Layer> layer = net[0]; // Get the first layer

Also, the output layer, i.e. the last added layer, can be accessed using the ``get_output_layer`` method:

.. code-block:: cpp

    std::shared_ptr<sim::Layer> output_layer = net.get_output_layer(); // Get the output layer

Finally, layers are iterable:

.. code-block:: cpp

    for (std::shared_ptr<sim::Layer> it : net) {
        // Do something
    }

.. note::
    Note that, when accessing layers, the based class ``Layer`` is returned.

Access and Mutate Weights
-------------------------

Synapses of layers are stored in ``NDArray`` objects. This object can be accessed using the ``get_weights`` method of a
layer. Taking the layer previously created, weights are accessed and mutated as follows:

.. code-block:: cpp

    sim::NDArray<float> &weights = layer->get_weights(); // Get the weight matrix of the layer.
    std::vector<unsigned int> dims = weights.get_dims(); // Get the dimensions of the matrix.

    float &w = weights.get(3, 5); // Get the weight of the connection.
    w = 0.1 // Set -0.1 to the connection.
    weights.set(-0.1, 3, 5); // Set -0.1 to the connection.

Here, we first get the weight matrix from the layer and its dimensions.
We then get the weight between the post-synaptic neuron at index 3
and the pre-synaptic neuron at index 5. The last two lines show two different ways to set a new value to the connection.

Alternatively, a contiguous and mutable vector can be obtained from the weight matrix:

.. code-block:: cpp

    sim::NDArray<float> &weights = layer->get_weights(); // Get the weight matrix of the layer.
    std::vector<unsigned int> dims = weights.get_dims(); // Get the dimensions of the matrix.

    sim::vector<float> &weights_cont = weights.get_values(); // Get weights as a vector.

    float w = weights[3 * dims[1] + 5] // Get the weight of the connection.
    weights_cont[3 * dims[1] + 5] = -0.1; // Set -0.1 to the connection.

This has the same effect as the previous code but it requires indexing using the dimensions of the matrix.

.. note::
    ``get_values`` returns a reference on a ``EvSpikeSim::vector`` object. In the CPU implementation,
    ``EvSpikeSim::vector`` is a standard ``std::vector``. In the GPU implementation, this is a ``std::vector``
    that uses a cuda managed pointer.

Run the SNN
-----------

After setting up the network, it is ready for inference. This is done by calling the ``infer`` method of
``SpikingNetwork`` with a **sorted** input ``SpikeArray``:

.. code-block:: cpp

    std::vector<unsigned int> input_indices = {1, 42, 21};
    std::vector<float> input_times = {0.030, 0.0017, 0.012};
    sim::SpikeArray input_spikes(input_indices, input_times); // Create input spikes

    input_spikes.sort(); // Sort spikes in time

    const sim::SpikeArray &output_spikes = net.infer(input_spikes); // Infer the network

.. note::
    Input spikes must be sorted in time before being sent for inference or ``infer`` will throw a runtime error.

Alternatively, input indices and times can directly be given as argument to ``infer``:

.. code-block:: cpp

    std::vector<unsigned int> input_indices = {1, 42, 21};
    std::vector<float> input_times = {0.030, 0.0017, 0.012};

    const sim::SpikeArray &output_spikes = net.infer(input_indices, input_times); // Infer the network

This way, input indices and times do not have to be sorted.

.. note::
    When passing indices and times as argument to ``infer``, a SpikeArray is implicitly created and sorted in time.

After inference, post-synaptic spikes of hidden layers can be accessed as follows:

.. code-block:: cpp

    std::shared_ptr<sim::Layer> &layer = net[0];
    const sim::SpikeArray &hidden_spikes = layer->get_post_spikes();

Additionally, neurons spike counts are also available:

.. code-block:: cpp

    std::shared_ptr<sim::Layer> &layer = net[0];
    const sim::vector<unsigned int> &hidden_spike_counts = layer->get_n_spikes();

Full Example
------------

.. code-block:: cpp

    #include <evspikesim/SpikingNetwork.h>
    #include <evspikesim/Layers/FCLayer.h>
    #include <evspikesim/Initializers/UniformInitializer.h>
    #include <evspikesim/Misc/RandomGenerator.h>

    namespace sim = EvSpikeSim;

    int main() {
        // Create network
        sim::SpikingNetwork network;

        // Layer parameters
        unsigned int n_inputs = 10;
        unsigned int n_neurons = 100;
        float tau_s = 0.010;
        float threshold = 0.1;

        // Uniform distribution for weight initialization (by default: [-1, 1])
        sim::RandomGenerator gen;
        sim::UniformInitializer init(gen);

        // Add fully-connected layer to the network
        std::shared_ptr<sim::FCLayer> layer = network.add_layer<sim::FCLayer>(n_inputs, n_neurons, tau_s, threshold, init);

        // Create input spikes
        std::vector<unsigned int> input_indices = {0, 8, 2, 4};
        std::vector<float> input_times = {0.010, 0.012, 0.21, 0.17};

        // Inference
        auto output_spikes = network.infer(input_indices, input_times);

        // Print output spikes
        std::cout << "Output spikes:" << std::endl;
        std::cout << output_spikes << std::endl;

        // Print output spike counts
        std::cout << "Output spike counts:" << std::endl;
        for (auto it : layer->get_n_spikes())
            std::cout << it << " ";
        std::cout << std::endl;
        return 0;
    }

EvSpikeSim Python
=================

In this section, we describe how to import EvSpikeSim to your Python project and create your first SNN.

Import EvSpikeSim
-----------------

We recommand to import EvSpikeSim as follows:

.. code-block:: python

    import evspikesim as sim

Spike
-----

Event-based simulations are driven by events. In EvSpikeSim, those events are the spikes that are received and fired
by neurons. Each spike event is represented by:

- its location;
- the time at which it occurs.

Such a representation of events is similar to the Address-Event Representation protocol (or AER) used for inter-chip
communication between neuromorphic hardware.

In EvSpikeSim, spikes are represented by the ``Spike`` class. This class has two attributes:

- An ``index`` attribute that is the index of the neuron that fired the spike, **in the layer where the event occured**. For example, ``index=2`` if the third neuron of the layer fired the spike.
- A ``time`` attribute that is the timing of the spike **in seconds**.

A spike can be instanciated as follows:

.. code-block:: python

    spike = sim.spike(42, 0.021)

This example, instanciates and construct a spike event at the neuron with index 42 and at time 0.021 seconds.

Moreover, spikes are comparable. By comparing spikes, only their timings are taken into account and not their indices.
All the standard comparators are available, i.e. ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``.

.. note::
    For the , ``==`` and , ``!=`` operators, spike timing are compared with a time precision of Spike::epsilon
    (by default: 1e-6, or 1 μs). See Python API documentation for more details.

Finally, spikes are printable:

.. code-block:: python

    print(spike)

Spike Arrays
------------

Spike trains are sequences of spikes. In EvSpikeSim, spike trains are represented by an array of ``Spike`` of type
``SpikeArray``. The ``SpikeArray`` effectively stores and sorts in time all the spikes of a layer, facilitating
iterations in chronological order during simulations.

To create a spike array, simply instanciate a new ``SpikeArray`` object as follows:

.. code-block:: python

    arr = sim.SpikeArray()

Alternatively, spike arrays can be created with a list of spikes indices and spike timings, such as:

.. code-block:: python

    arr = sim.SpikeArray(indices=[1, 42, 21], times=[0.030, 0.0017, 0.012])

or numpy ndarrays:

.. code-block:: python

    import numpy as np

    ...

    indices = np.array([1, 42, 21], dtype=np.uint32)
    times = np.array([0.030, 0.0017, 0.012], dtype=np.float32)
    arr = sim.SpikeArray(indices=indices, times=times);

which both directly populate the array with new ``Spike`` objects.

.. note::
    Numpy arrays of indices must be of dtype uint32 and arrays of timings must be of dtype float32.
    Numpy uses 64 bits values by default which is incompatible with the 32 bits values in EvSpikeSim.

.. todo::
    Fix the 64 bits incompatibility with numpy arrays.

To add a new spike, call the ``add`` method:

.. code-block:: python

    arr = sim.SpikeArray()

    arr.add(index=42, time=0.021) # Add new spike

The add method can also be called with lists of spikes indices and spike timings, such as:

.. code-block:: python

    arr = sim.SpikeArray()

    arr.add(indices=[1, 42, 21], times=[0.030, 0.0017, 0.012])  # Add new spikes

or with numpy arrays:

.. code-block:: python

    arr = sim.SpikeArray()

    indices = np.array([1, 42, 21], dtype=np.uint32)
    times = np.array([0.030, 0.0017, 0.012], dtype=np.float32)

    arr.add(indices=indices, times=times)  # Add new spikes

A spike array can be sorted in time by calling the ``sort`` method:

.. code-block:: python

    arr.sort()

Finally, spike arrays are also printable:

.. code-block:: python

    print(arr)

Create a Spiking Network
------------------------

SNNs in EvSpikeSim are feedforward networks. This means that layers are successively simulated and do not
form any cycle or recurrence.

To create a new (empty) SNN, instanciate a ``SpikingNetwork`` object:

.. code-block:: cpp

    net = sim.SpikingNetwork()

Add Layers of Spiking Neurons
------------------------------

.. note::
    Only a single type of layer (FCLayer) is currently available in EvSpikeSim. New type of layer, such as convolution,
    will be added in future releases.

So far, ``SpikingNetwork`` object is empty and requires layers to do meaniningful computation.
All layers in EvSpikeSim inherit from the ``Layer`` base class.
To add layer a Fully-Connected (FC) layer to it, call the ``add_fc_layer`` method with the following parameters:

.. code-block:: python

    net = sim.SpikingNetwork() # Network

    init = sim.initializers.UniformInitializer()
    net.add_fc_layer(n_inputs=100, n_neurons=1000, tau_s=0.010, threshold=0.1, initializer=init)

Several parameters are required for the construction of a FCLayer object:

- the number of input neurons, typically the number of neurons in the previous layer;
- the number of neurons in the layer;
- the synaptic time constant controlling the decay of the neurons;
- the threshold of the neurons;
- and a weight initializer.

Here, a new layer of 1000 fully-connected neurons receiving 100 inputs with a synaptic time constant of 10 milliseconds
and a threshold of 0.1 is added to the network. Its weights are initialized with a uniform distribution between -1 and 1
(i.e. the default lower and upper bounds of ``UniformInitializer``). Finally, the ``add_fc_layer`` method returns the
the newly created ``FCLayer`` object.

Access Layers
-------------

Layers in a spiking network can also be accessed as follows:

.. code-block:: python

    layer = net[0] # Get the first layer

Also, the output layer, i.e. the last added layer, can be accessed using the ``output_layer`` property:

.. code-block:: python

    output_layer = net.output_layer # Get the output layer

Finally, layers are iterable:

.. code-block:: python

    for layer in net:
        # Do something

.. note::
    Note that, when accessing layers, the based class ``Layer`` is returned.

Access and Mutate Weights
-------------------------

Synapses of layers accessed and mutated through a numpy ndarray. This object can be accessed using the ``weights``
attribute of a layer. Taking the layer previously created, weights are accessed and mutated as follows:

.. code-block:: python

    weights = layer.weights # Get the weight matrix of the layer.
    dims = layer.shape

    w = weights[3, 5] # Get the weight of the connection.
    weights[3, 5] = 0.1 # Set -0.1 to the connection.

Here, we first get the weight matrix from the layer and its dimensions.
We then get the weight between the post-synaptic neuron at index 3
and the pre-synaptic neuron at index 5. The last line shows how to set a new value to the same connection.

Alternatively, new weights can be set by setting the weights property with a numpy array:

.. code-block:: python

    new_weights = np.random.uniform(size=(1000, 100)) # Create new weights

    layer.weights = new_weights # Set weights

.. warning::
    When setting new weights, the numpy array **must** match the current shape of the weights. No check is performed
    by EvSpikeSim.

Run the SNN
-----------

After setting up the network, it is ready for inference. This is done by calling the ``infer`` method of
``SpikingNetwork`` with a **sorted** input ``SpikeArray``:

.. code-block:: python

    input_spikes = sim.SpikeArray(indices=[1, 42, 21], times=[0.030, 0.0017, 0.012]) # Create input spikes
    input_spikes.sort() # Sort spikes in time

    output_spikes = net.infer(input_spikes) # Infer the network

.. note::
    Input spikes must be sorted in time before being sent for inference or ``infer`` will throw an exception.

Alternatively, lists of input indices and times can directly be given as argument to ``infer``:

.. code-block:: python

    output_spikes = net.infer(indices=[1, 42, 21], times=[0.030, 0.0017, 0.012]) # Infer the network

or with numpy arrays:

.. code-block:: python

    indices = np.array([1, 42, 21], dtype=np.uint32)
    times = np.array([0.030, 0.0017, 0.012], dtype=np.float32)
    output_spikes = net.infer(indices=indices, times=times) # Infer the network

This way, input indices and times do not have to be sorted.

.. note::
    When passing indices and times as argument to ``infer``, a SpikeArray is implicitly created and sorted in time.

After inference, post-synaptic spikes of hidden layers can be accessed as follows:

.. code-block:: python

    layer = net[0]
    hidden_spikes = layer.post_spikes;

Additionally, neurons spike counts are also available:

.. code-block:: python

    layer = net[0];
    hidden_spike_counts = layer.n_spikes

Full Example
------------

.. code-block:: python

    import evspikesim as sim

    if __name__ == "__main__":
        # Create network
        network = sim.SpikingNetwork()

        # Uniform distribution for weight initialization (by default: [-1, 1])
        init = sim.initializers.UniformInitializer()

        # Add fully-connected layer to the network
        layer = network.add_fc_layer(n_inputs=10, n_neurons=100, tau_s=0.010, threshold=0.1, initializer=init)

        # Create input spikes
        input_indices = [0, 8, 2, 4]
        input_times = [1.0, 1.5, 1.2, 1.1]

        # Inference
        output_spikes = network.infer(input_indices, input_times)

        # Print output spikes
        print("Output spikes:")
        print(output_spikes)

        # Print output spike counts
        print("Output spike counts:")
        print(layer.n_spikes)
