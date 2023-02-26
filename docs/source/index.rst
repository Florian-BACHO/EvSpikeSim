=========================
EvSpikeSim (Experimental)
=========================

.. note::
    This project is under active development.
    Feature requests and bug reports are more than welcome.

Project Description
===================

EvSpikeSim is an experimental event-based Spiking Neural Networks (SNNs) simulator written in C++ for high performance and interfaced with Python.
This project aims to provide fast and accurate simulations of sparse SNNs for the development of training algorithms compatible with neuromorphic hardware.

Implemented Features
====================

- Fully-connected layers of Leaky Integrate-and-Fire (LIF) neurons
- Eligibility traces
- Simple Python3 interface compatible with numpy arrays
- Multi-theading on CPU
- NVIDIA GPU support

Neuron Model
============

The neuron model implemented in this simulator is the Current-Based Leaky Integrate-and-Fire (CuBa LIF) neuron. The membrane potential of each neuron `i` is defined as:

.. math::
    u_i(t) = \sum_{j} w_{i,j} \sum_{t_j} \underbrace{\left[\exp\left(\frac{t_j - t}{\tau}\right) - \exp\left(\frac{t_j - t}{\tau_s}\right) \right]}_{\text{Post-Synaptic Potential}} - \underbrace{\vartheta \sum_{t_i} \exp\left(\frac{t_i - t}{\tau}\right)}_{\text{Reset}}

where :math:`\tau_s` and :math:`\tau` are respectively the synaptic and membrane time constants, :math:`w_{i,j}` is the weight between the post-synaptic neuron :math:`i` and the pre-synaptic neuron :math:`j`, :math:`t_j < t` is a pre-synaptic pre-synaptic spike timings received at synapse :math:`j`, :math:`t_i < t` is a post-synaptic spike timing and :math:`\vartheta` is the threshold.

Pre-synaptic spikes are integrated over time with a double-exponential Post-Synaptic Potential kernel. When the membrane potential reaches its threshold, i.e. `u_i(t)=\vartheta`, a post-synaptic spike is emitted by the neuron `i` and the membrane potential is reset to zero.

.. warning::
    In EvSpikeSim, membrane time constants are constrained to twice the synaptic time constants, i.e. :math:`\tau = 2 \tau_s`. This allows us to isolate a closed-form solution for the spike time and achieve fast event-based inference without the use of numerical solvers.

.. toctree::
   :maxdepth: 1
   :caption: Table of Contents:

   self
   installation
   getting_started
   advanced_guide
   docker
   cpp_api/library_root
   python_api
   unit_tests