import csv
import math
from collections import Counter


def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset.append({
                'age': int(row['age']),
                'income': int(row['income']),
                'will_buy': int(row['will_buy'])
            })
    return dataset


def normalize_dataset(dataset):
    # Find min and max values for each feature
    ages = [data['age'] for data in dataset]
    incomes = [data['income'] for data in dataset]

    min_age, max_age = min(ages), max(ages)
    min_income, max_income = min(incomes), max(incomes)

    # Normalize each feature to 0-1 range
    for data in dataset:
        data['age'] = (data['age'] - min_age) / (max_age - min_age)
        data['income'] = (data['income'] - min_income) / (max_income - min_income)

    return dataset, (min_age, max_age, min_income, max_income)


def manhattan_distance(point1, point2):
    distance = 0
    distance += abs(point1['age'] - point2['age'])
    distance += abs(point1['income'] - point2['income'])
    return distance


def knn_predict(training_data, test_point, k=19):
    distances = []

    for train_point in training_data:
        dist = manhattan_distance(train_point, test_point)
        distances.append((dist, train_point['will_buy']))

    # Sort distances and get the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    # Get the class labels of the neighbors
    neighbor_classes = [neighbor[1] for neighbor in neighbors]

    # Return the most common class
    most_common = Counter(neighbor_classes).most_common(1)
    return most_common[0][0]


def train_test_split(dataset, test_size=0.2):
    split_idx = int(len(dataset) * (1 - test_size))
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    return train_set, test_set


def evaluate(y_true, y_pred):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    accuracy = correct / len(y_true)
    return accuracy


# Main execution
if __name__ == "__main__":
    # Load and prepare data
    dataset = load_dataset('Classification_dataset.csv')
    normalized_dataset, norm_params = normalize_dataset(dataset)

    # Split data
    train_set, test_set = train_test_split(normalized_dataset)

    # Make predictions
    predictions = []
    for test_point in test_set:
        # We don't want to include the actual label in the prediction
        test_point_copy = {'age': test_point['age'], 'income': test_point['income']}
        pred = knn_predict(train_set, test_point_copy, k=19)
        predictions.append(pred)

    # Get true labels
    true_labels = [point['will_buy'] for point in test_set]

    # Evaluate
    accuracy = evaluate(true_labels, predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Example prediction for a new data point
    new_point = {'age': 40, 'income': 50000}
    # Normalize the new point using the same parameters
    min_age, max_age, min_income, max_income = norm_params
    new_point_normalized = {
        'age': (new_point['age'] - min_age) / (max_age - min_age),
        'income': (new_point['income'] - min_income) / (max_income - min_income)
    }
    prediction = knn_predict(normalized_dataset, new_point_normalized, k=19)
    print(f"Prediction for age=40, income=50000: {
    'Will buy' if prediction == 1 else 'Wont buy'}")