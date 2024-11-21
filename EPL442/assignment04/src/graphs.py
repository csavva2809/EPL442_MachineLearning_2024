import numpy as np
import matplotlib.pyplot as plt

def plot_errors(results_file, output_file):
    # Load data and strip brackets if necessary
    with open(results_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
        data = []
        for line in lines:
            line = line.replace('[', '').replace(']', '')  # Remove brackets
            data.append([float(x) for x in line.split(',')])

    data = np.array(data)
    iterations = data[:, 0]
    training_errors = data[:, 1]
    testing_errors = data[:, 2]

    # Plot training and testing errors
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, training_errors, label='Training Error', marker='o')
    plt.plot(iterations, testing_errors, label='Testing Error', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Training and Testing Errors Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory

def plot_weights(weights_file, output_file):
    # Load weights from weights.txt
    weights = np.loadtxt(weights_file, delimiter=',')

    # Dynamically determine the shape for reshaping
    num_hidden_neurons = len(weights)  # Assume 1 output neuron
    num_output_neurons = 1  # You can adjust this if needed
    weights = weights.reshape((num_hidden_neurons, num_output_neurons))

    # Plot a heatmap of the weights
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Output Neurons')
    plt.ylabel('Hidden Neurons')
    plt.title('Heatmap of Weights')
    plt.savefig(output_file)  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory

if __name__ == "__main__":
    # Specify the output files
    results_file = "results.txt"
    weights_file = "weights.txt"

    # Specify the names for the PNG files
    errors_output_file = "errors_plot.png"
    weights_output_file = "weights_heatmap.png"

    # Generate and save the plots
    plot_errors(results_file, errors_output_file)
    plot_weights(weights_file, weights_output_file)

    print(f"Graphs saved as {errors_output_file} and {weights_output_file}")
