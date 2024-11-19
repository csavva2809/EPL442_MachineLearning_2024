# neural_network.py
import numpy as np
import random

np.random.seed(42)
random.seed(42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights and biases for the first hidden layer
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
        self.bias_hidden1 = np.zeros((1, hidden_size1))

        if hidden_size2 > 0:
            # Initialize weights and biases for the second hidden layer if specified
            self.weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
            self.bias_hidden2 = np.zeros((1, hidden_size2))
            # Output layer connects to the second hidden layer
            self.weights_hidden_output = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
        else:
            # Output layer connects directly to the first hidden layer
            self.weights_hidden_output = np.random.randn(hidden_size1, output_size) * np.sqrt(2.0 / hidden_size1)

        self.bias_output = np.zeros((1, output_size))

        # Initialize deltas for momentum
        self.delta_weights_ih1 = np.zeros_like(self.weights_input_hidden1)
        self.delta_bias_h1 = np.zeros_like(self.bias_hidden1)
        
        if hidden_size2 > 0:
            self.delta_weights_h1h2 = np.zeros_like(self.weights_hidden1_hidden2)
            self.delta_bias_h2 = np.zeros_like(self.bias_hidden2)
        
        self.delta_weights_ho = np.zeros_like(self.weights_hidden_output)
        self.delta_bias_o = np.zeros_like(self.bias_output)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.hidden_input1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden_output1 = self.relu(self.hidden_input1)
        
        if hasattr(self, 'weights_hidden1_hidden2'):
            # Forward pass through the second hidden layer if it exists
            self.hidden_input2 = np.dot(self.hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden2
            self.hidden_output2 = self.relu(self.hidden_input2)
            # Pass output of second hidden layer to the final layer
            self.final_input = np.dot(self.hidden_output2, self.weights_hidden_output) + self.bias_output
        else:
            # Directly pass the output of the first hidden layer to the final layer
            self.final_input = np.dot(self.hidden_output1, self.weights_hidden_output) + self.bias_output

        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backpropagation(self, X, y):
        output_errors = y - self.final_output
        output_delta = output_errors * self.sigmoid_derivative(self.final_output)

        if hasattr(self, 'weights_hidden1_hidden2'):
            hidden_errors2 = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta2 = hidden_errors2 * self.relu_derivative(self.hidden_output2)
            
            hidden_errors1 = np.dot(hidden_delta2, self.weights_hidden1_hidden2.T)
            hidden_delta1 = hidden_errors1 * self.relu_derivative(self.hidden_output1)

            # Update for second hidden layer
            self.delta_weights_h1h2 = self.learning_rate * np.dot(self.hidden_output1.T, hidden_delta2) + self.momentum * self.delta_weights_h1h2
            self.delta_bias_h2 = self.learning_rate * hidden_delta2.sum(axis=0, keepdims=True) + self.momentum * self.delta_bias_h2
            # Apply updates
            self.weights_hidden1_hidden2 += self.delta_weights_h1h2
            self.bias_hidden2 += self.delta_bias_h2
        else:
            # Only one hidden layer: calculate errors for that layer
            hidden_errors1 = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta1 = hidden_errors1 * self.relu_derivative(self.hidden_output1)

        # Momentum updates
        self.delta_weights_ho = self.learning_rate * np.dot(self.hidden_output2.T, output_delta) + self.momentum * self.delta_weights_ho if hasattr(self, 'weights_hidden1_hidden2') else self.learning_rate * np.dot(self.hidden_output1.T, output_delta) + self.momentum * self.delta_weights_ho
        self.delta_bias_o = self.learning_rate * output_delta.sum(axis=0, keepdims=True) + self.momentum * self.delta_bias_o

        self.delta_weights_ih1 = self.learning_rate * np.dot(X.T, hidden_delta1) + self.momentum * self.delta_weights_ih1
        self.delta_bias_h1 = self.learning_rate * hidden_delta1.sum(axis=0, keepdims=True) + self.momentum * self.delta_bias_h1

        # Apply updates
        self.weights_hidden_output += self.delta_weights_ho
        self.bias_output += self.delta_bias_o
        self.weights_input_hidden1 += self.delta_weights_ih1
        self.bias_hidden1 += self.delta_bias_h1

    def train(self, X, y, epochs, batch_size=32):
        with open("errors.txt", "w") as errors_file, open("successrate.txt", "w") as successrate_file:
            for epoch in range(epochs):
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                
                # Track total loss and accuracy for the entire epoch
                epoch_loss = 0
                epoch_correct = 0
                total_samples = 0
                
                for start in range(0, X.shape[0], batch_size):
                    end = start + batch_size
                    batch_indices = indices[start:end]
                    X_batch, y_batch = X[batch_indices], y[batch_indices]
                    self.feedforward(X_batch)
                    self.backpropagation(X_batch, y_batch)
                    
                    # Calculate batch loss and accuracy
                    batch_loss = np.mean(np.square(y_batch - self.final_output))
                    batch_correct = np.sum(np.argmax(self.final_output, axis=1) == np.argmax(y_batch, axis=1))
                    
                    # Accumulate epoch loss and accuracy
                    epoch_loss += batch_loss * len(y_batch)  # Multiply by batch size to get total loss for this batch
                    epoch_correct += batch_correct
                    total_samples += len(y_batch)
                
                # Average loss and accuracy for the epoch
                epoch_loss /= total_samples
                epoch_accuracy = epoch_correct / total_samples
                
                # Logging every 10 epochs for improved visibility
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy * 100:.2f}%")
                    errors_file.write(f"{epoch}\t{epoch_loss}\n")
                    successrate_file.write(f"{epoch}\t{epoch_accuracy * 100}\n")

    def predict(self, X):
        output = self.feedforward(X)
        return np.argmax(output, axis=1)
