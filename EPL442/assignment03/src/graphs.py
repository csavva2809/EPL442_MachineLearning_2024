import matplotlib.pyplot as plt
import numpy as np

# Διαβάζουμε τα δεδομένα από το results.txt
def load_results(file_path):
    epochs = []
    training_errors = []
    testing_errors = []

    with open(file_path, 'r') as file:
        next(file)  # Παράβλεψη της επικεφαλίδας
        for line in file:
            epoch, train_error, test_error = line.strip().split(',')
            epochs.append(int(epoch))
            training_errors.append(float(train_error))
            testing_errors.append(float(test_error))

    return np.array(epochs), np.array(training_errors), np.array(testing_errors)

# Διαβάζουμε τα δεδομένα από το clustering.txt
def load_clustering(file_path):
    positions = []
    labels = []

    with open(file_path, 'r') as file:
        next(file)  # Παράβλεψη της επικεφαλίδας
        for line in file:
            pos_label = line.strip().split('),')
            pos = pos_label[0].strip("()").split(',')
            label = pos_label[1].strip()
            positions.append((int(pos[0]), int(pos[1])))
            labels.append(label)

    return positions, labels

# Δημιουργία γραφήματος λάθους/εποχών
def plot_error_epoch(epochs, training_errors, testing_errors):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_errors, label='Training Error', marker='o')
    plt.plot(epochs, testing_errors, label='Testing Error', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("error_epoch_graph.png")
    plt.close()

# Δημιουργία πλέγματος SOM με ετικέτες
def plot_som_grid(positions, labels):
    grid_size = max(max(x for x, y in positions), max(y for x, y in positions)) + 1
    grid = [['' for _ in range(grid_size)] for _ in range(grid_size)]

    for (x, y), label in zip(positions, labels):
        grid[x][y] = label

    plt.figure(figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, grid_size - i - 1, grid[i][j], ha='center', va='center', fontsize=8)

    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.gca().invert_yaxis()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('SOM Clustering Grid')
    plt.savefig("som_grid.png")
    plt.close()

# Φόρτωση δεδομένων και δημιουργία γραφημάτων
epochs, training_errors, testing_errors = load_results("results.txt")
positions, labels = load_clustering("clustering.txt")

plot_error_epoch(epochs, training_errors, testing_errors)
plot_som_grid(positions, labels)

print("Οι εικόνες error_epoch_graph.png και som_grid.png δημιουργήθηκαν επιτυχώς!")
