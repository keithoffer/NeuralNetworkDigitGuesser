from neural_network import NeuralNetwork
import scipy.ndimage.interpolation as interpolation
import numpy as np

# http://makeyourownneuralnetwork.blogspot.com.au/2016/04/busting-past-98-accuracy.html

number_of_input_nodes = 28*28  # Input pictures are 28*28 pixels
number_of_hidden_nodes = 200
number_of_output_nodes = 10

learning_rate = 0.1

epochs = 5

n = NeuralNetwork(number_of_input_nodes, number_of_hidden_nodes, number_of_output_nodes, learning_rate)

with open('mnist_data/mnist_train.csv', 'r') as f:
    training_data_list = f.readlines()

for e in range(epochs):
    for line in training_data_list:
        all_values = line.split(',')
        targets = np.zeros(number_of_output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        # Rescale the input data to the range [0.01,1] from [0,255]
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        n.train(inputs, targets)
        # Augment the training data by additionally training on image rotated +10 degrees and - 10 degrees
        inputs_plus_10_degrees = interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, reshape=False).reshape(784)
        inputs_minus_10_degrees = interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, reshape=False).reshape(784)
        n.train(inputs_plus_10_degrees, targets)
        n.train(inputs_minus_10_degrees, targets)

    print(f"Finished epoch {e+1} of {epochs}")

n.save('neural_network.npz')
