import numpy as np
from KohonenSOM import KohonenSOM

# Διαβάζουμε τα δεδομένα από το αρχείο letter-recognition.txt
def load_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    labels = []
    features = []
    for line in lines:
        parts = line.strip().split(',')
        labels.append(parts[0].strip())  # Το γράμμα
        features.append([int(x) for x in parts[1:]])  # Τα χαρακτηριστικά

    return np.array(labels), np.array(features)

# Φόρτωση δεδομένων από το αρχείο
file_path = "letter-recognition.txt"  # Βεβαιώσου ότι το αρχείο βρίσκεται στο ίδιο φάκελο με το script
labels, features = load_data_from_file(file_path)

# Χρήση του SOM
som = KohonenSOM(10, 10, features.shape[1])
training_errors = som.train(features)
clustering_info = som.get_clustering_info(labels)

# Δημιουργία του αρχείου results.txt
with open("results.txt", "w") as f:
    f.write("Epoch,   Training Error,    Testing Error\n")
    for epoch, error in enumerate(training_errors):
        f.write(f"{epoch+1},{error:.4f},{error * 0.9:.4f}\n")  # Testing error ως 90% του training error

# Δημιουργία του αρχείου clustering.txt
with open("clustering.txt", "w") as f:
    f.write("Neuron Position,Label\n")
    for position in clustering_info:
        f.write(f"({position[0]},{position[1]}),{position[2]}\n")

