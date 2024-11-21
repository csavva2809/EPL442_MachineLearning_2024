import numpy as np

class RBFNetwork:
    def __init__(self, num_input_neurons, num_hidden_neurons, num_output_neurons, learning_rate, sigmas):
        self.num_input_neurons = num_input_neurons
        self.num_hidden_neurons = num_hidden_neurons
        self.num_output_neurons = num_output_neurons
        self.learning_rate = learning_rate

        # Ensure the length of sigmas matches the number of hidden neurons
        if len(sigmas) != num_hidden_neurons:
            raise ValueError(f"The length of sigmas ({len(sigmas)}) does not match the number of hidden neurons ({num_hidden_neurons}).")

        self.sigmas = np.array(sigmas)
        self.weights = np.random.randn(self.num_hidden_neurons, self.num_output_neurons)
        self.centers = np.zeros((self.num_hidden_neurons, self.num_input_neurons))

    def set_centers(self, centers):
        if len(centers) != self.num_hidden_neurons:
            raise ValueError("Number of centers must match the number of hidden neurons")
        self.centers = np.array(centers)

    def gaussian_rbf(self, x, center, sigma):
        distance = np.linalg.norm(x - center)
        return np.exp(- (distance ** 2) / (2 * sigma ** 2))
    
    def compute_hidden_layer_output(self, x):
        return np.array([self.gaussian_rbf(x, self.centers[i], self.sigmas[i]) for i in range(self.num_hidden_neurons)])
    
    def train(self, X_train, y_train, max_iterations, X_test=None, y_test=None):
        training_errors = []
        testing_errors = []

        for iteration in range(max_iterations):
            total_training_error = 0
            for x, y in zip(X_train, y_train):
                hidden_output = self.compute_hidden_layer_output(x)
                output = hidden_output @ self.weights
                error = y - output
                total_training_error += np.sum(error ** 2)
                self.weights += self.learning_rate * np.outer(hidden_output, error)

            avg_training_error = total_training_error / len(X_train)
            training_errors.append(avg_training_error)

            if X_test is not None and y_test is not None:
                total_testing_error = 0
                for x, y in zip(X_test, y_test):
                    hidden_output = self.compute_hidden_layer_output(x)
                    output = hidden_output @ self.weights
                    total_testing_error += (y - output) ** 2
                avg_testing_error = total_testing_error / len(X_test)
                testing_errors.append(avg_testing_error)

            print(f"Iteration {iteration + 1}/{max_iterations}, Training Error: {avg_training_error}")

        return training_errors, testing_errors

    def predict(self, X):
        predictions = []
        for x in X:
            hidden_output = self.compute_hidden_layer_output(x)
            output = hidden_output @ self.weights
            predictions.append(output)
        return np.array(predictions)
