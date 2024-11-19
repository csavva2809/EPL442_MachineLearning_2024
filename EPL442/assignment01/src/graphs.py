import matplotlib.pyplot as plt
import numpy as np
"""
Konstantinos Savva UC1058103

Usage:
pip install numpy
pip install matplotlib
python graphs.py / python3 graphs.py
"""
# Load errors.txt (iteration, training_error, testing_error)
errors = np.loadtxt('errors.txt')
iterations = errors[:, 0]
training_errors = errors[:, 1]
testing_errors = errors[:, 2]

# Plot training and testing errors
plt.plot(iterations, training_errors, label="Training Error")
plt.plot(iterations, testing_errors, label="Testing Error")
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Training and Testing Error over Time')
plt.legend()

# Save the figure as an image
plt.savefig('training_testing_errors.png')  # Saves the plot to a file
plt.show()  # Display the plot (optional)

# Load successrate.txt (iteration, training_success, testing_success)
success = np.loadtxt('successrate.txt')
training_success = success[:, 1]
testing_success = success[:, 2]

# Plot success rates
plt.figure()  # Create a new figure for success rate
plt.plot(iterations, training_success, label="Training Success")
plt.plot(iterations, testing_success, label="Testing Success")
plt.xlabel('Iterations')
plt.ylabel('Success Rate (%)')
plt.title('Training and Testing Success over Time')
plt.legend()

# Save the figure as an image
plt.savefig('training_testing_success_rate.png')  # Saves the plot to a file
plt.show()  # Display the plot (optional)
