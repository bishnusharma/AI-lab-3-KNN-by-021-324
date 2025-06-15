from sklearn.neighbors import KNeighborsClassifier

# Example dataset
X = [[1, 2], [2, 3], [3, 3], [6, 5], [7, 8]]
y = [0, 0, 0, 1, 1]

# Create model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Predict a new sample
sample = [[4, 4]]
prediction = model.predict(sample)
print("Predicted Class:", prediction[0])
