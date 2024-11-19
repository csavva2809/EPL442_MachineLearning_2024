import matplotlib.pyplot as plt
import pandas as pd

# Define file paths
error_file_path = 'errors.txt'  # Update if file path is different
success_file_path = 'successrate.txt'  # Update if file path is different

# Load data into DataFrames
error_data = pd.read_csv(error_file_path, sep="\t", header=None, names=["Epoch", "Loss"])
success_data = pd.read_csv(success_file_path, sep="\t", header=None, names=["Epoch", "Success Rate"])

# Plot and Save Training Loss over Epochs
plt.figure(figsize=(10, 5))
plt.plot(error_data["Epoch"], error_data["Loss"], label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")  # Save the plot as an image file
plt.close()  # Close the plot to save memory

# Plot and Save Training Success Rate over Epochs
plt.figure(figsize=(10, 5))
plt.plot(success_data["Epoch"], success_data["Success Rate"], label="Training Success Rate", color="green")
plt.xlabel("Epoch")
plt.ylabel("Success Rate (%)")
plt.title("Training Success Rate Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("training_success_rate.png")  # Save the plot as an image file
plt.close()  # Close the plot to save memory

