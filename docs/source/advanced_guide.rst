==============
Advanced Guide
==============

This guide introduces advanced concepts that are implemented in EvSpikeSim.

.. contents:: Table of Contents

Spike Buffers
=============

In EvSpikeSim, neurons are processed in parallel. Therefore, post-synaptic spikes need to be stored in buffers during
the inference. If the buffer of a neuron becomes full, all buffers are processed into the post-synaptic spike array
and the inference is resumed.
However, there is no known method to predict the number of post-synaptic spikes that will be fired by
a neuron. These buffers thus have a fixed size of 64 (floats) per neuron by default.
The size of the buffer impacts both the memory usage and the performance.
Large buffers use more memory but require fewer inference iterations than small buffers which can slow down
simulations. When creating a layer, you can specify the size of the spike buffer to use.

.. rubric:: EvSpikeSim C++

In EvSpikeSim C++, this is done by giving setting the last argument of the ``add_layer`` method:

.. code-block:: cpp

    unsigned int buffer_size = 32;
    std::shared_ptr<sim::FCLayer> layer = net.add_layer<sim::FCLayer>(n_inputs, n_neurons, tau_s, threshold, init, buffer_size);

.. rubric:: EvSpikeSim Python

In EvSpikeSim C++, this is done by giving setting the ``buffer_size`` argument:

.. code-block:: python

    net.add_fc_layer(n_inputs=100, n_neurons=1000, tau_s=0.010, threshold=0.1, initializer=init, buffer_size=32)

Eligibility Traces
==================

Eligibility traces are temporal records of past events that are essential for learning.
They are typically represented by ordinary differential equations of the form:

.. math::
    \tau \frac{\mathrm{d}s(t)}{\mathrm{d}t} = -s(t)

When a post-synaptic or pre-synaptic events occur, alterations of eligibility traces can take place.
For example the following update can be performed when an event occur at a time :math:`t^\prime`:

.. math::
    s(t^\prime) \leftarrow s(t^\prime) + 1

The trace decays with a time constant :math:\tau, which results in a fading memory of the past activity, where greater
importance is given to recent events over those that occurred farther in the past.

Therefore, an eligibility trace updated with the previous example would corresponds to a simple low-pass filter
of past events, such as:

.. math::
    s(t) = \sum_{t^\prime < t} \exp\left(\frac{t^\prime - t}{\tau}\right)

The following figure illustrates the behavior of this trace.

.. plot::
    :caption: An example of a simple low-pass filter eligibility trace with a time constant of 3ms receiving two events at times 1ms and 4ms.
    :align: center

    import matplotlib.pyplot as plt
    import numpy as np

    def exp_kernel(t_prime, t, tau):
        return 0.0 if t < t_prime else np.exp((t_prime - t) / tau)

    t_1 = 0.001
    t_2 = 0.004
    tau = 0.003

    x = np.arange(0, 0.020, 1e-4)
    y = [exp_kernel(t_1, t, tau) + exp_kernel(t_2, t, tau) for t in x]

    plt.figure(figsize=(4,3))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.plot(x, y)
    plt.xlabel("Time (second)")
    plt.ylabel("s(t)")

Eligibility Traces Template
---------------------------

In EvSpikeSim, eligibility traces and their behavior are defined in external C++ files (.cpp) that are compiled
when creating layers. The default template for a source file is the following:

.. code-block:: cpp

    #include <evspikesim/Layers/EligibilityTraces.h>

    namespace sim = EvSpikeSim;

    sim::vector<float> sim::synaptic_traces_tau(float tau_s, float tau) {
        return {};
    }

    sim::vector<float> sim::neuron_traces_tau(float tau) {
        return {};
    }

    CALLBACK void sim::on_pre_neuron(float weight, float *neuron_traces) {

    }

    CALLBACK void sim::on_pre_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {

    }

    CALLBACK void sim::on_post_neuron(float *neuron_traces) {

    }

    CALLBACK void sim::on_post_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {

    }

.. note::
    The ``CALLBACK`` decorator needs to be added in front of the ``on_pre_neuron``, ``on_pre_synapse``,
    ``on_post_neuron`` and ``on_post_synapse`` callback functions. It allows cross-compatibility for both CPU and GPU
    implementations.

The following sections describe how to define eligibility traces and their behavior using these functions, and
explain how to create layers using source files.

Define Eligibility Traces
-------------------------

Eligibility traces are present at two levels in neurons:

- at the synapse level (i.e. *synaptic traces*)
- at the neuron level (i.e. *neuron traces*).

The two functions synaptic_traces_tau and neuron_traces_tau are used to define the time constants of traces at the
synaptic and neuron levels, respectively. These functions must return vectors of time constants **in second**.

The number of time constants returned by each function defines the number of eligibility traces that will be
created at each corresponding level. For example, the following example creates two traces **per synapse** and a
single eligibility trace **per neuron**:

.. code-block:: cpp

    sim::vector<float> sim::synaptic_traces_tau(float tau_s, float tau) {
        return {1.1 * tau, INFINITY};
    }

    sim::vector<float> sim::neuron_traces_tau(float tau) {
        return {0.9 * tau};
    }

The ``tau_s`` and ``tau`` arguments are the synaptic and membrane time constants that are given when creating a layer.
You can use them, as in this example, to create traces time constants that are relative to the neuron model time
constants.

.. note::
    ``INFINITY`` is used to disable decay. Traces with an infinite time constant act as non-leaky integrators and can
    be used to accumulate information.

Update Eligibility Traces
-------------------------

Traces can be updated and **locally** interact with each other when events occur. More precisely, a neuron trace
can be updated:

- when a pre-synaptic spike is received at **any** of the neuron's synapse, using the ``on_pre_neuron`` callback function
- when a post-synaptic spike is fired by the neuron, using the ``on_post_neuron`` callback function.

On the other hand, synaptic traces can be updated:

- when a pre-synaptic spike is received at the **same** synapse, using the ``on_pre_synapse`` callback function
- when a post-synaptic spike is fired by the neuron, using the ``on_post_synapse`` callback function.

.. note::
    The decay of traces is internally managed by the simulator.

The order in which the callback are called respect the direction of propagation of the information. When a pre-synaptic
spike occurs, decay is first applied to the eligibiliity traces, the ``on_pre_synapse`` is called for the corresponding
synapse of the neuron, then the ``on_pre_neuron`` is called for the neuron that received the spike. The order of call at
pre-synaptic events is summarized in the following diagram:

.. graphviz::

   digraph {
        rankdir="LR";
        "Pre-synaptic spike" -> "Traces decay" -> "on_pre_synapse" -> "on_pre_neuron";

        "Pre-synaptic spike"[shape=polygon, sides=4, fontsize=11]
        "Traces decay"[shape=polygon, sides=4, fontsize=11]
        "on_pre_synapse"[shape=polygon, sides=4, fontsize=11]
        "on_pre_neuron"[shape=polygon, sides=4, fontsize=11]
   }

|

When a post-synaptic spike is fired by a neuron, decay is first applied to the eligibiliity traces,
the ``on_post_neuron`` is called for the neuron that fired the spike, then `on_post_synapse`` is called
**for every synapse** of the neuron. The order of call at post-synaptic events is summarized in the following diagram:

.. graphviz::

   digraph {
        rankdir="LR";
        "Post-synaptic spike" -> "Traces decay" -> "on_post_neuron" -> "on_post_synapse (all)";

        "Post-synaptic spike"[shape=polygon, sides=4, fontsize=11]
        "Traces decay"[shape=polygon, sides=4, fontsize=11]
        "on_post_neuron"[shape=polygon, sides=4, fontsize=11]
        "on_post_synapse (all)"[shape=polygon, sides=4, fontsize=11]
   }

|

in the next sub-sections, we describe how to use each update callback.

.. rubric:: on_pre_synapse

The ``on_pre_synapse`` function receives three arguments:

- the weight of the synapse that received the pre-synaptic spike
- the (immutable) neuron traces
- the traces of the synapse that received the spike.

For example, the following callback updates the second trace of the synapse with the first trace of the neuron, scaled
by the weight:

.. code-block:: cpp

    CALLBACK void sim::on_pre_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
        synaptic_traces[1] += weight * neuron_traces[0];
    }

.. rubric:: on_pre_neuron

The ``on_pre_neuron`` function receives two arguments:

- the weight of the synapse that received the pre-synaptic spike
- the neuron traces.

The following example callback updates the first neuron trace by integrating the weight of the synapse that received
the spike:

.. code-block:: cpp

    CALLBACK void sim::on_pre_neuron(float weight, float *neuron_traces) {
        neuron_traces[0] += weight;
    }

.. rubric:: on_post_synapse

The ``on_post_synapse`` function receives three arguments:

- the weight of the synapse that is being updated
- the (immutable) neuron traces
- the traces of the synapse that is being udpated

.. note::
    Note that, unlike ``on_pre_synapse``, the ``on_post_synapse`` callback is called **for every** synapse at
    each post-synaptic event.

For example, the following callback updates the second trace of the synapse with the first trace of the neuron, scaled
by the weight:

.. code-block:: cpp

    CALLBACK void sim::on_post_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
        synaptic_traces[1] += weight * neuron_traces[0];
    }

.. rubric:: on_post_neuron

The ``on_post_neuron`` function only receives the neuron traces as argument.

The following callback example increases the first neuron trace when a post-synaptic spike occurs:

.. code-block:: cpp

    CALLBACK void sim::on_post_neuron(float *neuron_traces) {
        neuron_traces[0] += 1.0;
    }

.. warning::
    Keep in mind that these callbacks functions are called every times an event occurs.
    Try to keep these functions as simple as possible to avoid any computational overhead that would slow down
    simulations.

Full Example: Spike Time-Dependant Plasticity (STDP)
----------------------------------------------------

In Spike Time-Dependant Plasticity (STDP), the strength of a synapse is modified based on the timing of the spikes.
If the pre-synaptic neuron fires just before the post-synaptic neuron, the strength of the synapse is increased.
Conversely, if the post-synaptic neuron fires just before the pre-synaptic neuron, the strength of the synapse is
decreased.

Formally, the change of weight :math:`\Delta w_{i,j}` induced by STDP between the pre-synaptic neuron :math:`j` and the
post-synaptic neuron :math:`i` is defined as a sum over all pre-synaptic and post-synaptic spike times, such as:

.. math::
    \Delta w_{i,j} = \sum_{t_{\text{post}}} \sum_{t_{\text{pre}}} \text{STDP}\left( t_{\text{post}} - t_{\text{pre}} \right)

where

.. math::
    \text{STDP}\left( \Delta t \right) = \begin{cases}
        a_{\text{pre}} \exp\left ( \frac{-\Delta t}{\tau_{\text{pre}}} \right ) & \text{ if } \Delta t > 0 \\
        -a_{\text{post}} \exp\left ( \frac{\Delta t}{\tau_{\text{post}}} \right ) & \text{ if } \Delta t < 0
    \end{cases}

This kernel induces Long Term Potentiation (LTP) with an amplitude :math:`a_{\text{pre}}` when a pre-synaptic spike occurs
**before** a post-synaptic spike. If a pre-synaptic spike occurs **after** a post-synaptic spike,
Long Term Depression (LTD) is induced with an amplitude :math:`a_{\text{post}}`. Here, :math:`\tau_{\text{pre}}` and
:math:`\tau_{\text{post}}` represent the time constants of the LTP and LTD respectively.
The following figure illustrates the behavior of this kernel as a function of the temporal difference between
pre-synaptic and post-synaptic spikes:

.. plot::
    :caption: Long Term Potentiation (LTP) and Long Term Depression (LTD) induced by STDP.
    :align: center

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['text.usetex'] = True

    def stdp_kernel(delta_t, tau):
        return -np.exp(delta_t / tau) if delta_t < 0 else np.exp(-delta_t / tau)

    tau = 0.003

    x = np.arange(-0.010, 0.010, 1e-4)
    y = [stdp_kernel(delta_t, tau) for delta_t in x]

    plt.figure(figsize=(4,3))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.plot(x, y)

    ax = plt.gca()
    ax.axhline(0, linestyle='--', color='gray')
    ax.axvline(0, linestyle='--', color='gray')

    plt.xlabel(r"$\Delta t$ (second)")
    plt.ylabel(r"$STDP\left( \Delta t \right)$")

To implement STDP with eligibility traces, we need to define one neuron trace :math:`s_{\text{post}}(t)` that keeps
track of the post-synaptic activity and two synaptic traces :math:`s_{\text{pre}}(t)` and  :math:`\Delta w(t)` that
respectively keeps track of the pre-synaptic activity and the changes of weight.

These traces have the following linear dynamics:

.. math::
    \tau_{\text{post}} \frac{\mathrm{d}s_{\text{post}}(t)}{\mathrm{d}t} =& -s_{\text{post}}(t) \\
    \tau_{\text{pre}} \frac{\mathrm{d}s_{\text{pre}}(t)}{\mathrm{d}t} =& -s_{\text{pre}}(t) \\
    \frac{\mathrm{d}\Delta w(t)}{\mathrm{d}t} =& 0

When a pre-synaptic spike occurs, the traces are updated as follows:

.. math::
    s_{\text{pre}}(t) \leftarrow& s_{\text{pre}}(t) + a_{\text{pre}}(t) \\
    \Delta w(t) \leftarrow& \Delta w(t) - s_{\text{post}}(t)

And when a post-synaptic spike is fired by the neuron:

.. math::
    s_{\text{post}}(t) \leftarrow& s_{\text{post}}(t) + a_{\text{post}}(t) \\
    \Delta w(t) \leftarrow& \Delta w(t) + s_{\text{pre}}(t)

Using our template of callbacks, STDP is implemented as:

.. code-block:: cpp

    #include <evspikesim/Layers/EligibilityTraces.h>

    namespace sim = EvSpikeSim;

    static constexpr float a_pre = 1.0;
    static constexpr float a_post = 1.0;

    sim::vector<float> sim::synaptic_traces_tau(float tau_s, float tau) {
        return {tau, INFINITY};
    }

    sim::vector<float> sim::neuron_traces_tau(float tau) {
        return {tau};
    }

    CALLBACK void sim::on_pre_neuron(float weight, float *neuron_traces) {

    }

    CALLBACK void sim::on_pre_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
        synaptic_traces[0] += a_pre;
        synaptic_traces[1] -= neuron_traces[0];
    }

    CALLBACK void sim::on_post_neuron(float *neuron_traces) {
        neuron_traces[0] += a_post;
    }

    CALLBACK void sim::on_post_synapse(float weight, const float *neuron_traces, float *synaptic_traces) {
        synaptic_traces[1] += synaptic_traces[0];
    }

Create a Layer with Eligibility Traces
--------------------------------------

Now that we have defined our eligibility traces, we can create layers with them. When creating a layer, EvSpikeSim
will compile the given source file and load it as a dynamic library. If the file has already been loaded,
EvSpikeSim will skip the compilation.

.. rubric:: EvSpikeSim C++

In EvSpikeSim C++, layers with eligibility traces are created as follows:

.. code-block:: cpp

    std::shared_ptr<sim::FCLayer> layer = network.add_layer_from_source<FCLayer>("path/to/source/file.cpp", n_inputs, n_neurons, tau_s, threshold, init, buffer_size);

.. rubric:: EvSpikeSim Python

In EvSpikeSim Python, layers with eligibility traces are created as follows:

.. code-block:: python

    layer = network.add_fc_layer_from_source("path/to/source/file.cpp", n_inputs, n_neurons, tau_s, threshold, init, buffer_size)

Access Eligibility Traces
-------------------------

After inference, eligibility traces are available. Their values correspond to their last update (i.e. last event that occured).

.. rubric:: EvSpikeSim C++

In EvSpikeSim C++, eligibility traces are accessed as follows:

.. code-block:: cpp

    const sim::vector<float> &neuron_traces = layer->get_neuron_traces();
    const sim::vector<float> &synaptic_traces = layer->get_synaptic_traces();

.. rubric:: EvSpikeSim Python

In EvSpikeSim Python, eligibility traces are accessed as follows:

.. code-block:: python

    neuron_traces = layer.neuron_traces
    synaptic_traces = layer.synaptic_traces

Random and Reproducibility
==========================

To allow reeproducibility, it is possible to set the seed of random generators.

.. rubric:: EvSpikeSim C++

In EvSpikeSim C++, this can be done by providing the seed at the construction of a ``RandomGenerator`` object:

.. code-block:: cpp

    unsigned long seed = 42;
    sim::RandomGenerator gen(seed);

.. rubric:: EvSpikeSim Python

In EvSpikeSim Python, the random generator used by initializers is global. To set its seed, proceed as follows:

.. code-block:: python

    seed = 42
    sim.random.set_seed(seed)

