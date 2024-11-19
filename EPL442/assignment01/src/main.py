import numpy as np
"""
Konstantinos Savva UC1058103

Usage:
pip install numpy
python main.py / python3 main.py
"""
class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate, momentum):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Randomly initialize weights and biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_neurons, self.hidden_neurons))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_neurons, self.output_neurons))
        
        self.bias_hidden = np.zeros((1, self.hidden_neurons))
        self.bias_output = np.zeros((1, self.output_neurons))

        # Momentum terms
        self.delta_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.delta_weights_hidden_output = np.zeros_like(self.weights_hidden_output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagate(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_output = self.sigmoid(self.output_input)
        
        return self.output_output

    def back_propagate(self, inputs, expected_output):
        output_error = expected_output - self.output_output
        output_delta = output_error * self.sigmoid_derivative(self.output_output)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights with momentum
        self.delta_weights_hidden_output = (self.learning_rate * np.dot(self.hidden_output.T, output_delta)) + (self.momentum * self.delta_weights_hidden_output)
        self.weights_hidden_output += self.delta_weights_hidden_output

        self.delta_weights_input_hidden = (self.learning_rate * np.dot(inputs.T, hidden_delta)) + (self.momentum * self.delta_weights_input_hidden)
        self.weights_input_hidden += self.delta_weights_input_hidden

        # Update biases
        self.bias_output += self.learning_rate * np.sum(output_delta, axis=0)
        self.bias_hidden += self.learning_rate * np.sum(hidden_delta, axis=0)

    def train(self, train_inputs, train_outputs, test_inputs, test_outputs, max_iterations):
        # Open files to log errors and success rates
        with open('errors.txt', 'w') as errors_file, open('successrate.txt', 'w') as success_file:
            for iteration in range(max_iterations):
                # Forward propagation on training data
                self.forward_propagate(train_inputs)

                # Back propagation for weight updates
                self.back_propagate(train_inputs, train_outputs)

                # Training error (mean squared error)
                training_error = np.mean(np.square(train_outputs - self.output_output))

                # Forward propagate test data (to calculate testing error and success)
                test_output = self.forward_propagate(test_inputs)
                testing_error = np.mean(np.square(test_outputs - test_output))

                # Calculate success rates for training and testing data
                train_success_rate = np.mean((self.output_output > 0.5) == train_outputs) * 100
                test_success_rate = np.mean((test_output > 0.5) == test_outputs) * 100

                # Log errors
                errors_file.write(f"{iteration+1} {training_error} {testing_error}\n")
                
                # Log success rates
                success_file.write(f"{iteration+1} {train_success_rate} {test_success_rate}\n")

                # Print the iteration and error for tracking
                print(f"Iteration {iteration+1}, Training Error: {training_error}, Testing Error: {testing_error}, Training Success: {train_success_rate}%, Testing Success: {test_success_rate}%")

# Load data from file
def load_data(file_name):
    data = np.loadtxt(file_name)
    inputs = data[:, :-1]
    outputs = data[:, -1].reshape(-1, 1)
    return inputs, outputs

# Main function to set up the network and train it
def main():
    with open('parameters.txt', 'r') as file:
        parameters = dict(line.strip().split() for line in file)
    
    # Convert necessary parameters to integers or floats
    input_neurons = int(parameters['numInputNeurons'])
    hidden_neurons = int(parameters['numHiddenLayerOneNeurons'])
    output_neurons = int(parameters['numOutputNeurons'])
    learning_rate = float(parameters['learningRate'])
    momentum = float(parameters['momentum'])
    max_iterations = int(parameters['maxIterations'])

    # Load training and test data
    train_inputs, train_outputs = load_data(parameters['trainFile'])
    test_inputs, test_outputs = load_data(parameters['testFile'])

    # Create the neural network
    nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons, learning_rate, momentum)

    # Train the neural network and create log files
    nn.train(train_inputs, train_outputs, test_inputs, test_outputs, max_iterations)

if __name__ == "__main__":
    main()
