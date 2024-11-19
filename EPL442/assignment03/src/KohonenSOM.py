import numpy as np
from tqdm import tqdm

# Ορισμός κλάσης για το Kohonen SOM
class KohonenSOM:
    def __init__(self, m, n, dim, learning_rate=0.02, epochs=150):
        self.m = m  # Γραμμές του SOM πλέγματος
        self.n = n  # Στήλες του SOM πλέγματος
        self.dim = dim  # Διάσταση του διανύσματος εισόδου
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(m, n, dim)  # Τυχαία αρχικοποίηση βαρών

    def train(self, data):
        training_errors = []
        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            epoch_error = 0
            for sample in data:
                # Υπολογισμός νευρώνα νικητή
                winner = self.find_winner(sample)
                # Υπολογισμός λάθους
                epoch_error += np.linalg.norm(self.weights[winner] - sample)
                # Ενημέρωση βαρών για τον νικητή
                self.update_weights(winner, sample)
            # Μείωση του learning rate
            self.learning_rate *= 0.95
            training_errors.append(epoch_error / len(data))
        
        return training_errors

    def find_winner(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return np.unravel_index(np.argmin(distances), (self.m, self.n))

    def update_weights(self, winner, sample):
        for i in range(self.m):
            for j in range(self.n):
                distance = np.linalg.norm(np.array([i, j]) - np.array(winner))
                influence = np.exp(-distance)
                self.weights[i, j] += influence * self.learning_rate * (sample - self.weights[i, j])

    def get_clustering_info(self, labels):
        clustering_info = []
        for i in range(self.m):
            for j in range(self.n):
                clustering_info.append((i, j, labels[np.random.randint(0, len(labels))]))  # Παράδειγμα για το γράμμα
        return clustering_info