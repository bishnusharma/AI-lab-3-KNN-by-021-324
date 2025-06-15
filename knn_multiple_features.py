from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Sample dataset
data = [
    {"name": "Mad Max", "action": 40, "dialogues": 20, "duration": 2.5, "violence": 9, "rating": 8.1, "genre": 1},
    {"name": "John Wick", "action": 35, "dialogues": 25, "duration": 3.0, "violence": 8, "rating": 7.4, "genre": 1},
    {"name": "The Godfather", "action": 10, "dialogues": 85, "duration": 6.0, "violence": 3, "rating": 9.2, "genre": 0},
    {"name": "Forrest Gump", "action": 5, "dialogues": 91, "duration": 6.5, "violence": 1, "rating": 8.8, "genre": 0},
    {"name": "Die Hard", "action": 38, "dialogues": 30, "duration": 3.5, "violence": 7, "rating": 7.9, "genre": 1},
    {"name": "The Pursuit of Happiness", "action": 3, "dialogues": 94, "duration": 6.8, "violence": 1, "rating": 8.0, "genre": 0},
]

# Create DataFrame
df = pd.DataFrame(data)

# Select features and label
X = df[["action", "dialogues", "duration", "violence", "rating"]]
y = df["genre"]

# Initialize and train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# New movie data (with feature names as columns)
new_movie_df = pd.DataFrame([[18, 60, 4.5, 4, 7.5]], columns=X.columns)

# Predict genre
prediction = model.predict(new_movie_df)
genre = "Action" if prediction[0] == 1 else "Drama"

print("📽️ Predicted Genre:", genre)
