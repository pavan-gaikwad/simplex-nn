import random
import math

# this is a simple implementation of a Multi Layer Perceptron
# I am using a tanh activation function here.
# Idea is to keep the code readable and not use any libraries as far as possible
# Goal with this code is to minimise the error on each example in the input data set - there is no batching implemented
#


class Neuron:
    def __init__(self, incoming):
        self.incoming = incoming
        self.out = None
        self.bias = 0
        self.grad = 0
        # set activation function to tanh
        self.activation = lambda x: (
            math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        # set the tanh derivate as the derivative function
        self.activation_derivative = lambda x: 1 - x**2

        # initialise with random weights
        if self.incoming:
            self.weights = [random.uniform(-1, 1) for _ in self.incoming]
        else:
            self.weights = [random.uniform(-1, 1)]

    def update_weights(self, learning_rate):
        new_weights = []
        for weight in self.weights:
            new_weights.append(weight + learning_rate * self.grad)
        self.weights = new_weights
        self.bias += learning_rate * self.grad

    def run(self, values=None):

        self.init_inputs = values
        if values:

            self.out = self.activation(
                sum([wi*xi for xi, wi in zip(values, self.weights)], self.bias))
        else:

            self.out = self.activation(
                sum([wi*xi.out for xi, wi in zip(self.incoming, self.weights)], self.bias))

    def __repr__(self):
        return "output: %s | grad: %s " % (self.out, self.grad)


class MLP:
    def __init__(self, inputs, outputs, hidden_layer_counts, output_count):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layer_counts = hidden_layer_counts
        self.nn = None
        self.output_count = output_count

    def construct(self):
        layers = []
        layers.append([Neuron(None) for _ in self.inputs[0]])
        for h in self.hidden_layer_counts:
            layers.append([Neuron(layers[-1]) for _ in range(h)])
        layers.append([Neuron(layers[-1]) for _ in range(self.output_count)])
        self.nn = layers

    # def calc_loss(self, target):
    #     loss = (target-self.nn[-1][0].out)**2
    #     self.loss = loss

    def forward(self, x):
        # for every example provided
        for i in range(len(x)):
            # get first layer
            j = 0
            for neuron in self.nn[0]:
                neuron.run([x[i][j]])
                j += 1

            for layer in self.nn[1:]:
                for neuron in layer:
                    neuron.run()

    def backward(self, output, learning_rate):

        # set gradients
        # get output layer
        i = 0  # variable to track outputs
        for neuron in self.nn[-1]:
            # calculate the derivative of error w.r.t. output
            dedy = 2 * (output[i] - neuron.out)
            # calculate the derivative of output w.r.t. activation function

            dyda = neuron.activation_derivative(neuron.out)
            for input, weight in zip(neuron.weights, neuron.incoming):
                # calculate the derivative of the output w.r.t. weighted sum
                dzdw = input
                # calculate the gradient
                neuron.grad += dedy * dyda * dzdw
            # update the gradient in all the incoming neurons
            for incoming_neuron in neuron.incoming:
                incoming_neuron.grad += neuron.grad
            i += 1

        # get hidden layers
        for i in range(len(self.nn) - 2, -1, -1):
            layer = self.nn[i]
            for neuron in layer:

                # ignore input layer
                if neuron.incoming:
                    self_grad = 0
                    # calculate the derivative of output w.r.t. activation function
                    dyda = neuron.activation_derivative(neuron.out)
                    for input, weight in zip(neuron.weights, neuron.incoming):
                        # calculate the derivative of the output w.r.t. weighted sum
                        dzdw = input
                        # calculate the gradient
                        self_grad += dyda * dzdw

                    neuron.grad = self_grad * neuron.grad

                    # update the gradient in all the incoming neurons
                    for incoming_neuron in neuron.incoming:
                        incoming_neuron.grad += neuron.grad

        # get input layer

        for neuron in self.nn[0]:
            self_grad = 0
            # calculate the derivative of output w.r.t. activation function
            dyda = neuron.activation_derivative(neuron.out)
            for input, weight in zip(neuron.weights, neuron.init_inputs):
                # calculate the derivative of the output w.r.t. weighted sum
                dzdw = input
                # calculate the gradient
                self_grad += dyda * dzdw

            neuron.grad = self_grad * neuron.grad

        # update weights
        for layer in self.nn:
            for neuron in layer:
                neuron.update_weights(learning_rate)

    def reset_grad(self):
        for layer in self.nn:
            for neuron in layer:
                neuron.grad = 0

    def __repr__(self):
        return str(self.nn)


# code to test run the neural network
x = [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7], [0.4, 0.8, 0.2], [0.11, 0.13, 0.9]]
y = [0.2, 0.1, 0.2, 0.1]


def train(x, y):
    losses = []

    # define a Multi Layer Perceptron with above inputs, 2 hidden layers of 6 neurons each and 1 output layer
    m = MLP(x, y, [6, 6], 1)

    # create a MLP structure and initialise the neurons
    m.construct()

    iterations = 10000
    learning_rate = 0.0001
    for i in range(len(x)):
        for _ in range(iterations):
            # reset all gradients before running a feed-forward
            m.reset_grad()
            m.forward([x[i]])

            # run back propogation for the input
            m.backward([y[i]], learning_rate)

            # once the error is less than 0.01, stop the training for the current example and move on to the next one.
            if abs(y[i]-m.nn[-1][0].out) < 0.01:
                break

        # display the output ( this only works for this example with 1 output)
        print(m.nn[-1][0].out)
    return m


def infer(model, input):
    model.forward(input)
    print("Output is: %f" % model.nn[-1][0].out)


model = train(x, y)

input = [[0.2, 0.8, 0.4]]
infer(model, input)
