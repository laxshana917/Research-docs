# Simple KNN Example for Beginners
# Step 1: Import the tools we need
# We use 'sklearn' which contains the KNN algorithm
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Create some imaginary data (The "Training" Data)
# Imagine we have fruit data: [Weight in grams, Texture (1=smooth, 10=rough)]
# X is our data (features)
X = [
    [150, 1],  # Fruit 1: 150g, Smooth (Apple)
    [170, 1],  # Fruit 2: 170g, Smooth (Apple)
    [140, 1],  # Fruit 3: 140g, Smooth (Apple)
    [130, 10], # Fruit 4: 130g, Rough (Orange)
    [150, 9],  # Fruit 5: 150g, Rough (Orange)
    [140, 8]   # Fruit 6: 140g, Rough (Orange)
]

# y is our labels (answers)
y = ['Apple', 'Apple', 'Apple', 'Orange', 'Orange', 'Orange']

# Step 3: Create the KNN Classifier
# n_neighbors=3 means "look at the 3 closest neighbors to decide"
knn = KNeighborsClassifier(n_neighbors=3)

# Step 4: Train the model
# This is where the computer "memorizes" the data
knn.fit(X, y)
print("Model trained! It has memorized the fruit data.")

# Step 5: Make a prediction
# We have a new mystery fruit: 145g and Texture 2 (Smooth)
mystery_fruit = [[145, 2]] 

# Ask the model to predict what it is
prediction = knn.predict(mystery_fruit)

print(f"The mystery fruit (145g, smooth) is predicted to be an: {prediction[0]}")

# Let's try another one: 135g, Texture 9 (Rough)
mystery_fruit_2 = [[135, 9]]
prediction_2 = knn.predict(mystery_fruit_2)

print(f"The second mystery fruit (135g, rough) is predicted to be an: {prediction_2[0]}")
