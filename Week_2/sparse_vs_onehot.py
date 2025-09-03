import numpy as np
import matplotlib.pyplot as plt

# Example: 1 million labels, 10 classes
num_labels = 1_000_000
num_classes = 10

# Sparse representation (just store class indices)
sparse_labels = np.random.randint(0, num_classes, size=num_labels)

# One-hot representation
one_hot_labels = np.eye(num_classes)[sparse_labels]

print("Sparse shape:", sparse_labels.shape)
print("One-hot shape:", one_hot_labels.shape)

# Calculate memory usage
sparse_size = sparse_labels.nbytes
one_hot_size = one_hot_labels.nbytes

print("Sparse size:", sparse_size, "bytes")
print("One-hot size:", one_hot_size, "bytes")

# Plot bar chart
plt.figure(figsize=(6,4))
plt.bar(["Sparse Encoding", "One-Hot Encoding"], [sparse_size, one_hot_size], color=["skyblue", "salmon"])
plt.ylabel("Memory (bytes)")
plt.title("Memory Usage: Sparse vs One-Hot Encoding")
plt.grid(True)
plt.show()
