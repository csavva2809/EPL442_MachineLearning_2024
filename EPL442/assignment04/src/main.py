from RBFNetwork import RBFNetwork
from data_utils import read_parameters, load_data, load_centers, save_results, save_weights
import numpy as np

# Load parameters
params = read_parameters('parameters.txt')

# Extract parameters
num_hidden_neurons = params['numHiddenLayerNeurons']
num_input_neurons = params['numInputNeurons']
num_output_neurons = params['numOutputNeurons']
learning_rate = params['learningRates']
sigmas = params['sigmas']
max_iterations = params['maxIterations']

# Load data
X_train, y_train, X_test, y_test = load_data(params['trainFile'], params['testFile'])

# Adjust the number of centers dynamically
np.random.seed(42)  # For reproducibility
indices = np.random.choice(X_train.shape[0], num_hidden_neurons, replace=False)
centers = X_train[indices]

# Initialize and set up the RBF Network
rbf_network = RBFNetwork(num_input_neurons, num_hidden_neurons, num_output_neurons, learning_rate, sigmas)
rbf_network.set_centers(centers)

# Train the network
training_errors, testing_errors = rbf_network.train(X_train, y_train, max_iterations, X_test, y_test)

# Save results
save_results("results.txt", training_errors, testing_errors)
save_weights("weights.txt", rbf_network.weights)

print("Training complete. Results and weights have been saved.")
