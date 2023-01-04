import numpy as	np
import evspikesim as sim

# Network definition                                                                                
network = sim.Network()
network.add_fc_layer(2, 3, 0.020, 0.020 * 0.1)
network[0].weights = np.array([[-1.0, -2.0],
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
