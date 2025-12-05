# ===== Project: Diagnosing Bias vs Variance with Learning Curves=====

import numpy as np
import matplotlib.pyplot as plt

# Given dataset sizes
m = np.array([50, 100, 200, 500, 1000])

# Given errors (simulated)
training_error = np.array([0.25, 0.22, 0.20, 0.18, 0.17])
validation_error = np.array([0.30, 0.28, 0.27, 0.26, 0.25])

# ============================
# Plot the learning curves
# ============================
plt.plot(m, training_error, label="Training Error")
plt.plot(m, validation_error, label="Validation Error")
plt.xlabel("Training Set Size (m)")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()


# ============================
# Determine if the model suffers from:
# - High Bias
# - High Variance
# - Or neither
# ============================

diagnosis = "High Bias" 


# ============================
# Suggest one improvement  
# Examples:
#   - "Add more features"
#   - "Add regularization"
#   - "Get more data"
# ============================

improvement = "Use a more complex model or add more features"


# ============================
# Output results
# ============================

print("Model Diagnosis:", diagnosis)
print("Recommended Improvement:", improvement)
