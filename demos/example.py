import evspikesim as sim
import numpy as np


if __name__ == "__main__":
    # Create network
    network = sim.SpikingNetwork()

    # Uniform initial distribution (by default: [-1, 1])
    init = sim.initializers.UniformInitializer()

    # Add fully-connected layer to the network
    layer = network.add_fc_layer(n_inputs=3, n_neurons=30, tau_s=0.020, threshold=0.1, initializer=init)

    # Create input spikes
    input_indices = np.array([0, 1, 2, 1], dtype=np.int32)
    input_times = np.array([1.0, 1.5, 1.2, 1.1], dtype=np.float32)

    # Inference
    output_spikes = network.infer(input_indices, input_times)

    print("Output spikes:")
    print(output_spikes)

    print("Output spike counts:")
    print(layer.n_spikes)
