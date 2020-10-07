from random import random
import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        self.neural_network = list()
        self.last_layer_neurons = 0
    
    def add_layer(self, num_neurons, activation = 'sigmoid', input_layer = False):
        if not input_layer:
            self.neural_network.append([{'bias': random(), 
                                         'weights': [random() for i in range(self.last_layer_neurons)],
                                         'activation': activation} 
                                         for i in range(num_neurons)])
        self.last_layer_neurons = num_neurons

    def sigmoid(self, value, derivative = False):
        if derivative:
            return value * (1 - value)
        return 1/(1 + np.exp(-1 * value))
    
    def relu(self, value, derivative = False):
        if derivative:
            return 1 if value > 0 else 0
        return value if value > 0 else 0
    
    def activate(self, inputs, weights, bias, activation):
        activation_functions = {'sigmoid': self.sigmoid, 'relu': self.relu}
        activation_function = activation_functions[activation]

        return activation_function(sum([inputs[i] * weights[i] for i in range(len(inputs))]) + bias)
    
    def forward_propagate(self, inputs):
        for layer in self.neural_network:
            layer_outputs = list()
            for neuron in layer:
                neuron['output'] = self.activate(inputs, neuron['weights'], neuron['bias'], neuron['activation'])
                layer_outputs.append(neuron['output'])
            inputs = layer_outputs
        return inputs
    
    def print_attribute(self, key):
        attribute = []
        for layer in self.neural_network:
            attribute.append([neuron[key] for neuron in layer])
        print(attribute)
    
    def backward_propagate(self, inputs, expected_outputs, learning_rate):
        for i in reversed(range(len(self.neural_network))):
            layer = self.neural_network[i]
            layer_errors = list()
            if i == len(self.neural_network) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron_error = expected_outputs[j] - neuron['output']
                    layer_errors.append(neuron_error)
            else:
                for j in range(len(layer)):
                    neuron_error = sum([neuron['weights'][j] * neuron['error'] for neuron in self.neural_network[i + 1]])
                    layer_errors.append(neuron_error)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['error'] = layer_errors[j] * self.sigmoid(neuron['output'], True)
                
        for i in range(len(self.neural_network)):
            layer = self.neural_network[i]
            for neuron in layer:
                for j in range(len(neuron['weights'])):
                    neuron['weights'][j] += learning_rate * inputs[j] * neuron['error']
                neuron['bias'] += learning_rate * neuron['error']
            inputs = [neuron['output'] for neuron in layer]
    
    def train(self, x_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]
                outputs = self.forward_propagate(x)
                total_error += sum([(y[j] - outputs[j]) ** 2 for j in range(len(outputs))])
                self.backward_propagate(x, y, learning_rate)
            print("Epoch: {}, Total Error: {:.3f}".format(epoch, total_error))
    
    def predict(self, x_test, y_test):
        num_correct = 0
        num_wrong = 0
        for i in range(len(x_test)):
            x = x_test[i]
            y = y_test[i]
            outputs = self.forward_propagate(x)
            outputs = 1 if outputs[0] > .5 else 0 
            if outputs == y[0]:
                num_correct += 1
            else:
                num_wrong += 1
        print("Correct: {}, Incorrect: {}, Accuracy: {}".format(num_correct, num_wrong, num_correct/(num_correct + num_wrong)))
