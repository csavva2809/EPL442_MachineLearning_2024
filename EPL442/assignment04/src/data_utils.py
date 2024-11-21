import numpy as np

def read_parameters(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            # Check if the value is a list
            if ',' in value:
                parameters[key] = [float(v) for v in value.split(',')]
            # Check if the value is a number (int or float)
            elif value.replace('.', '', 1).isdigit():
                parameters[key] = float(value) if '.' in value else int(value)
            # Otherwise, treat it as a string (e.g., file paths)
            else:
                parameters[key] = value
    return parameters


def load_data(train_file, test_file):
    X_train = np.loadtxt(train_file, delimiter=',')
    y_train = X_train[:, -1]
    X_train = X_train[:, :-1]
    X_test = np.loadtxt(test_file, delimiter=',')
    y_test = X_test[:, -1]
    X_test = X_test[:, :-1]
    return X_train, y_train, X_test, y_test

def load_centers(centres_file):
    return np.loadtxt(centres_file, delimiter=',')

def normalize_data(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

def save_results(results_file, training_errors, testing_errors):
    with open(results_file, 'w') as file:
        file.write("Iteration,Training Error,Testing Error\n")
        for i, (train_error, test_error) in enumerate(zip(training_errors, testing_errors)):
            file.write(f"{i+1},{float(train_error)},{float(test_error)}\n")

def save_weights(weights_file, weights):
    np.savetxt(weights_file, weights, delimiter=',')
