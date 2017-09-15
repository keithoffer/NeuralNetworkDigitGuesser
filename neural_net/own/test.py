from neural_network import NeuralNetwork
import numpy as np

n = NeuralNetwork(load_from_file='neural_network.npz')

with open('mnist_data/mnist_test.csv', 'r') as f:
    test_data_list = f.readlines()

scorecard = []
for record in test_data_list:
    all_values = record.split(',')

    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)

    outputs = n.query(inputs)
    label = np.argmax(outputs)

    scorecard.append(int(label == correct_label))

scorecard_array = np.array(scorecard)
print("performance = ", scorecard_array.mean())
