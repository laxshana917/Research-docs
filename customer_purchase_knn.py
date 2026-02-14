# Real-World Scenario: Customer Purchase Prediction
# Problem: A car company wants to know if a new customer will buy their Luxury SUV.
# Data: We have data on past customers (Age, Estimated Salary) and if they bought it (0=No, 1=Yes).

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler # CRITICAL for real-world KNN

# Step 1: Create the Data (Age, Salary)
# In the real world, you would load this from a CSV file (e.g., pd.read_csv('sales.csv'))
X_train = [
    [20, 20000], [25, 30000], [30, 40000], # Young, Low Salary -> No
    [22, 50000], [28, 60000],              # Young, Med Salary -> No
    [45, 90000], [50, 100000],             # Older, High Salary -> Yes
    [35, 120000], [40, 130000],            # Mid-age, High Salary -> Yes
    [55, 30000], [60, 25000],              # Older, Low Salary -> Yes (Retired?)
    [25, 80000]                            # Young, High Salary -> Yes
]

# 0 = Didn't Buy, 1 = Bought
y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1] 

print("--- Data Created ---")
print(f"We have {len(X_train)} past customer records.")

# Step 2: Feature Scaling (The "Real World" Trick)
# KNN uses distance. SAlary (100,000) is HUGE compared to Age (40).
# If we don't scale, Salary will dominate. Distance will depend only on salary.
# We use StandardScaler to make both features contribute equally.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learn the scale and adjust data

# Step 3: Train the Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
print("Model Trained on SCALED data!")

# Step 4: The New Customer Scenario
# A new customer walks in.
# Age: 30
# Estimated Salary: $87,000
new_customer = [[30, 87000]]

# CRITICAL: We must scale the new customer's data using the SAME scaler as before!
new_customer_scaled = scaler.transform(new_customer)

# Step 5: Predict
prediction = knn.predict(new_customer_scaled)
result = "Will Buy" if prediction[0] == 1 else "Will Not Buy"

print("\n--- Prediction ---")
print(f"New Customer (Age: 30, Salary: $87,000)")
print(f"Prediction: {result}")

# Optional: Let's see who the neighbors were
distances, indices = knn.kneighbors(new_customer_scaled)
print("\n--- Why? (The Nearest Neighbors) ---")
for i in indices[0]:
    neighbor_features = X_train[i]
    neighbor_label = "Bought" if y_train[i] == 1 else "Did Not Buy"
    print(f"Neighbor: Age {neighbor_features[0]}, Salary ${neighbor_features[1]} -> {neighbor_label}")
