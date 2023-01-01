import sys
import numpy as np
import evspikesim as sim

def main():
    np.random.seed(42) # Set seeds for reproducibility
    sim.set_seed(42)
    np.set_printoptions(threshold=sys.maxsize)

    # Inputs
    n_inputs = 100
    n_input_spikes = 30
    max_input_spike_time = 0.010 # 10 ms
    # Random input generation
    input_indices = np.random.randint(0, n_inputs, size=(n_input_spikes,),
                                      dtype=np.uint32) # Type MUST be specified
    input_times = np.random.uniform(0, max_input_spike_time,
                                    size=(n_input_spikes,)).astype(np.float32) # Type MUST be specified

    # Network definition
    n_neurons_hidden_1 = 1024
    n_neurons_hidden_2 = 512
    n_neurons_out = 10
    tau_s = 0.010 # Synaptic time constant of 10 ms. Membrane time constant is tau = 2 * tau_s = 20 ms 
    threshold_hidden_1 = 0.5 * tau_s
    threshold_hidden_2 = 8.0 * tau_s
    threshold_out = 4.0 * tau_s

    # Network instanciation
    network = sim.Network()

    # Create layers
    network.add_fc_layer(n_inputs, n_neurons_hidden_1, tau_s, threshold_hidden_1)
    network.add_fc_layer(n_neurons_hidden_1, n_neurons_hidden_2, tau_s, threshold_hidden_2)
    network.add_fc_layer(n_neurons_hidden_2, n_neurons_out, tau_s, threshold_out)

    print("Input spikes:")
    print(input_indices)
    print(input_times)
    
    # Inference
    network.infer(input_indices, input_times)

    
    output_indices, output_times = network.output_layer.spikes
    print("Output spikes:")
    print(output_indices)
    print(output_times)
    print("Hidden layer 1 spike counts:")
    print(network[0].spike_counts)
    print("Hidden layer 2 spike counts:")
    print(network[1].spike_counts)
    print("Output spike counts:")
    print(network.output_layer.spike_counts)

if __name__ == "__main__":
    main()
