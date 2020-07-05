import numpy as np
import sys

class NeuralNetwork():
    
    def __init__(self):
        # Set synaptic weights to a 10x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((10, 1)) - 1

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x*(1-x)

    def train(self, training_inputs, training_outputs, n):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for iteration in range(n):
            # Pass training set through the neural network
            output = self.normalize(training_inputs)

            # Calculates the error b/w estimated and actual output
            error = training_outputs - output

            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def normalize(self, inputs):
        """
        Takes user inputs, applies weights, and passes into sigmoid function
        """
        inputs = inputs.astype(float)
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))



if __name__ == "__main__":

    # Initialize the neural network
    neural_network = NeuralNetwork()

    # Print the initial, random synaptic weights
    sys.stdout.write("Random starting synaptic weights:\n{}".format(neural_network.synaptic_weights))
    sys.stdout.flush()

    # The training set, 11 examples of 10 binary values
    training_inputs = np.array([[0,0,1,0,0,1,0,1,0,1],
                                [1,1,1,1,0,1,0,1,1,0],
                                [1,0,1,1,1,0,1,0,1,1],
                                [1,1,1,0,0,0,0,0,0,1],
                                [0,1,1,1,0,1,0,0,1,1],
                                [1,0,1,1,1,0,0,0,1,0],
                                [0,1,0,0,0,1,1,1,0,0],
                                [0,0,1,1,1,0,0,1,0,0],
                                [0,0,0,0,0,0,0,0,1,0],
                                [1,1,0,0,0,1,0,1,0,1],
                                [0,0,0,0,0,0,0,0,0,0]
                                ])
    # The outputs that correspond to each training input example
    # Note, the outputs are simply the fourth value in the input
    training_outputs = np.array([[0,1,1,0,1,1,0,1,0,0,0]]).T

    # The number of training iterations
    n = 100000

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, n)

    sys.stdout.write("Synaptic weights after training:\n{}\n\n".format(neural_network.synaptic_weights))
    sys.stdout.flush()

    # asks for inputs for a new test case
    abc = []
    for idx in range(len(training_inputs[0])):
        idx += 1 # Count from 1-10 instead of 0-9
        curr_input_val = str(input("Input {} (Enter 0 or 1): ".format(idx))) # asks for user input
        abc.append(curr_input_val) # appends user input to array of inputs
    abc = np.array(abc) # converts all the inputs to an numpy array


    sys.stdout.write("\nNew case from user: input data = {}\nOutput estimate:\n{}\n\n".format(
        abc,neural_network.normalize(abc)))
    sys.stdout.flush()

    # Note: may not work with all 0's because of the sigmoid function,
    # but it works well for any combination of 1's and 0's