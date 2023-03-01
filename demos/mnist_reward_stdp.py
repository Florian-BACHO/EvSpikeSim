import timeit

from mnist import MNIST
import evspikesim as sim
import numpy as np

# Requires python-mnist:
# pip3 install python-mnist

MNIST_PATH = "datasets/mnist/"

ENCODING_WINDOW = 0.010
MAX_PIXEL_VALUE = 255
N_INPUTS = 28 * 28

N_LABELS = 10
TAU_S = 0.010
THRESHOLD = 5.0

TARGET_FALSE = 5
TARGET_TRUE = 30

LEARNING_RATE = 3e-4

def image_to_spikes(x):
    '''
    Converts the input image into arrays of spike indices and spike timings with a temporal coding.
    :param x: The input image.
    :return: An array of encoded spikes.
    '''
    all_times = (1 - x / MAX_PIXEL_VALUE) * ENCODING_WINDOW

    spike_mask = x != 0
    indices = np.arange(len(x))[spike_mask]
    times = all_times[spike_mask]

    out = sim.SpikeArray(indices.astype(np.uint32), times.astype(np.float32))
    out.sort()
    return out


def convert_all_images(images):
    '''
    Convert all the given images into spikes with a temporal coding.
    :param images: A list of images to convert
    :return: A list of spike arrays.
    '''
    return [image_to_spikes(x) for x in images]

class MSELoss:
    def __init__(self, target_false, target_true):
        self.target_false = target_false
        self.target_true = target_true

    def compute_loss_and_errors(self, spike_counts, label):
        targets = np.full(spike_counts.shape, self.target_false)
        targets[label] = self.target_true

        errors = targets - spike_counts
        loss = np.sum(np.square(errors))

        return loss, errors

    def decode(self, spike_counts):
        return np.argmax(spike_counts)

if __name__ == "__main__":
    print("Loading dataset.")

    dataset = MNIST(MNIST_PATH)
    train_x, train_y = dataset.load_training()
    test_x, test_y = dataset.load_testing()

    train_indices = np.arange(len(train_x))
    np.random.shuffle(train_indices)
    train_x = np.array(train_x)[train_indices]
    train_y = np.array(train_y)[train_indices]

    print("Converting images to spikes.")

    train_x = convert_all_images(train_x)
    test_x = convert_all_images(np.array(test_x))

    print("Creating network.")

    init = sim.initializers.UniformInitializer(lower_bound=-1.0, upper_bound=1.0)

    net = sim.SpikingNetwork()
    #output_layer = net.add_fc_layer_from_source("callbacks/RewardSTDP.cpp", N_INPUTS, N_LABELS, TAU_S, THRESHOLD, init, buffer_size=64)
    net.add_fc_layer(N_INPUTS, 800, TAU_S, 0.1, init, buffer_size=64)
    net.add_fc_layer(800, 800, TAU_S, 10.0, init, buffer_size=64)
    output_layer = net.add_fc_layer(800, N_LABELS, TAU_S, 10.0, init, buffer_size=64)

    loss_fct = MSELoss(TARGET_FALSE, TARGET_TRUE)

    print()
    print("Training started.")
    cumum_loss = 0
    cumul_hit = 0
    for i, (x, y) in enumerate(zip(train_x, train_y)):
        print(timeit.timeit(lambda: net.infer(x), number=1000))
        exit()

        n_output_spikes = output_layer.n_spikes
        loss, errors = loss_fct.compute_loss_and_errors(n_output_spikes, y)
        pred = loss_fct.decode(n_output_spikes)
        hit = pred == y
        cumum_loss += loss
        cumul_hit += hit

        """print(hidden1_layer.n_spikes)
        print(hidden2_layer.n_spikes)
        print(n_output_spikes)
        exit()"""
        #if n_output_spikes[y] == 0:
        #    print(f"SILENT LABEL: {y}")

        #grad = output_layer.synaptic_traces[..., 1] * np.expand_dims(errors, 1)

        #output_layer.weights -= LEARNING_RATE * grad

        if i % 10 == 0:
            print(f"{i} / 60000 | Loss: {cumum_loss / (i + 1)}, Accuracy: {100 * cumul_hit / (i + 1)}")

    print()
    print("Testing started.")
    cumum_loss = 0
    cumul_hit = 0
    for i, (x, y) in enumerate(zip(test_x, test_y)):
        net.infer(x)

        n_output_spikes = output_layer.n_spikes
        loss, _ = loss_fct.compute_loss_and_errors(n_output_spikes, y)
        pred = loss_fct.decode(n_output_spikes)
        hit = pred == y
        cumum_loss += loss
        cumul_hit += hit

        if i % 1000 == 0:
            print(f"{i} / 10000 | Loss: {cumum_loss / (i + 1)}, Accuracy: {100 * cumul_hit / (i + 1)}")

    print(f"Testing metrics | Loss: {cumum_loss / 10000}, Accuracy: {100 * cumul_hit / 10000}")