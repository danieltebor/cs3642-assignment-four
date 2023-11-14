import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# Features and labels.
_features = None
_labels = None
_encoded_labels = None

try:
    # Modified from https://archive.ics.uci.edu/dataset/53/iris
    # Fetch iris dataset from UCI repository.
    iris = fetch_ucirepo(id=53) 

    # Features and labels.
    # Features is a numpy array of shape (150, 4).
    _features = iris.data.features.values
    # Labels is a numpy array of shape (150, 1).
    _labels = iris.data.targets.values.ravel()

    # Map labels to integers.
    # Sklearn's LabelEncoder is used to encode labels with value between 0 and n_classes-1.
    # This effectively maps the labels to integers as so:
    # 'Iris-setosa' -> 0
    # 'Iris-versicolor' -> 1
    # 'Iris-virginica' -> 2
    label_encoder = LabelEncoder()
    _encoded_labels = label_encoder.fit_transform(_labels)

except Exception as e:
    print(f'Failed to fetch iris dataset from UCI repository: {e}')
    print('Using local iris dataset.')

    # Fetch iris dataset from local file.
    bezdekIris = pd.read_csv('./data/bezdekIris.data', header=None)
    iris = pd.read_csv('./data/iris.data', header=None)

    # Concatenate the data
    data = pd.concat([bezdekIris, iris])

    # Features and labels
    _features = data.iloc[:, :-1].values
    _labels = data.iloc[:, -1].values.ravel()

    # Map labels to integers
    label_encoder = LabelEncoder()
    _encoded_labels = label_encoder.fit_transform(_labels)

# Split data 50-50 into training and validation sets.
# train_test_split is a convenience function to split data into training and validation sets.
TRAINING_FEATURES, VALIDATION_FEATURES, TRAINING_LABELS, VALIDATION_LABELS = train_test_split(_features, _encoded_labels, test_size=0.5, random_state=0)