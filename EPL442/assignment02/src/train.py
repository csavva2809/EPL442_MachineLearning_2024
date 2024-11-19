# train.py
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load parameters
def load_parameters(filename):
    params = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split()
            try:
                params[key] = float(value) if '.' in value else int(value)
            except ValueError:
                params[key] = value  # Keep as string if not numeric
    return params

# Prepare data
def load_data(filepath):
    data = pd.read_csv(filepath, header=None)
    data.columns = ['label'] + [f'feature_{i}' for i in range(16)]
    for i in range(16):
        data[f'feature_{i}'] = data[f'feature_{i}'] / 15.0
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    X = data.drop('label', axis=1).values
    y = np.eye(26)[data['label'].values]  # One-hot encoding
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Load parameters and data
parameters = load_parameters('parameters.txt')
X_train, X_test, y_train, y_test = load_data(parameters['trainFile'])

# Initialize the network with parameters
nn = NeuralNetwork(
    input_size=parameters['numInputNeurons'],
    hidden_size1=parameters['numHiddenLayerOneNeurons'],
    hidden_size2=parameters['numHiddenLayerTwoNeurons'],
    output_size=parameters['numOutputNeurons'],
    learning_rate=parameters['learningRate'],
    momentum=parameters['momentum']
)

# Train the network
nn.train(X_train, y_train, epochs=parameters['maxIterations'], batch_size=32)
