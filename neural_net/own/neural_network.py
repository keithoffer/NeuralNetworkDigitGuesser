import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes=10, hidden_nodes=10, output_nodes=10, learning_rate=0.3, load_from_file=None):
        if load_from_file is None:
            self.number_of_input_nodes = input_nodes
            self.number_of_hidden_nodes = hidden_nodes
            self.number_of_output_nodes = output_nodes
            self.learning_rate = learning_rate

            # Link weight matrices
            # Sampling randomly from a Normal distribution, mean 0, std -1/sqrt(# of incoming links)
            self.weights_input_hidden = np.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
            self.weights_hidden_output = np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))
        else:
            with open(load_from_file, 'rb') as f:
                npzfile = np.load(f)
                self.weights_input_hidden = npzfile['weights_input_hidden']
                self.weights_hidden_output = npzfile['weights_hidden_output']
                self.number_of_hidden_nodes = npzfile['number_of_hidden_nodes']
                self.number_of_input_nodes = npzfile['number_of_input_nodes']
                self.number_of_output_nodes = npzfile['number_of_output_nodes']

        # The activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # Remember we need to set ndmin to 2 so we don't have a 1D array (can't transpose it then)
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = self.weights_input_hidden @ inputs
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = self.weights_hidden_output @ hidden_outputs
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = self.weights_hidden_output.T @ output_errors

        # update the weights for the links between the hidden and output layers
        self.weights_hidden_output += self.learning_rate * (
            output_errors * final_outputs * (1.0 - final_outputs)) @ hidden_outputs.T
        # update the weights for the links between the input and hidden layers
        self.weights_input_hidden += self.learning_rate * (
            hidden_errors * hidden_outputs * (1.0 - hidden_outputs)) @ inputs.T

    def query(self, inputs_list):
        # Remember we need to set ndmin to 2 so we don't have a 1D array (can't transpose it then)
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = self.weights_input_hidden @ inputs
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = self.weights_hidden_output @ hidden_outputs
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f, weights_input_hidden=self.weights_input_hidden,
                     weights_hidden_output=self.weights_hidden_output,
                     number_of_input_nodes=self.number_of_input_nodes,
                     number_of_output_nodes=self.number_of_output_nodes,
                     number_of_hidden_nodes=self.number_of_hidden_nodes,
                     learning_rate=self.learning_rate)
