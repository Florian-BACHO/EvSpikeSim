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
