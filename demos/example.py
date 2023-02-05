import evspikesim as sim
import numpy as np


if __name__ == "__main__":
    # Create network
    network = sim.SpikingNetwork()

    # Add fully-connected layer to the network
    desc = sim.layers.FCLayerDescriptor(n_inputs=2, n_neurons=3, tau_s=0.020, threshold=0.020 * 0.2)
    layer = network.add_layer(desc)

    # Set weights
    layer.weights = np.array([[1.0, 0.3],
                              [-0.1, 0.8],
                              [0.5, 0.4]], dtype=np.float32)

    # Mutate weight
    layer.weights[0, 1] -= 0.1

    # Create input spikes
    input_indices = np.array([0, 1, 1], dtype=np.int32)
    input_times = np.array([1.0, 1.5, 1.2], dtype=np.float32)

    # Inference
    output_spikes = network.infer(input_indices, input_times)

    print("Output spikes:")
    print(output_spikes)

    print("Output spike counts:")
    print(layer.n_spikes)
