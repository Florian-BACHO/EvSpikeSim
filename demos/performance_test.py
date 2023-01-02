import sys
import numpy as np
import evspikesim as sim
from timeit import timeit

def infer(network, input_indices, input_times):
    network.reset()
    network.infer(input_indices, input_times)

def main():
    np.random.seed(42) # Set seeds for reproducibility
    sim.set_seed(42)
    np.set_printoptions(threshold=sys.maxsize)

    # Inputs
    n_inputs = 1024
    n_input_spikes = 100
    max_input_spike_time = 0.010 # 10 ms
    # Random input generation
    input_indices = np.random.randint(0, n_inputs, size=(n_input_spikes,),
                                      dtype=np.uint32) # Type MUST be specified
    input_times = np.random.uniform(0, max_input_spike_time,
                                    size=(n_input_spikes,)).astype(np.float32) # Type MUST be specified

    # Network definition
    n_neurons_hidden_1 = 1024
    n_neurons_hidden_2 = 1024
    tau_s = 0.010 # Synaptic time constant of 10 ms. Membrane time constant is tau = 2 * tau_s = 20 ms 
    threshold_hidden_1 = 1.0 * tau_s
    threshold_hidden_2 = 8.0 * tau_s
    threshold_out = 4.0 * tau_s

    # Network instanciation
    network = sim.Network()

    # Create layers
    network.add_fc_layer(n_inputs, n_neurons_hidden_1, tau_s, threshold_hidden_1)
    #network.add_fc_layer(n_neurons_hidden_1, n_neurons_hidden_2, tau_s, threshold_hidden_2)

    network.reset()
    network.infer(input_indices, input_times)
    print(network[0].spike_counts.sum())
    
    print(timeit(lambda: infer(network, input_indices, input_times), number=10))
    
if __name__ == "__main__":
    main()
