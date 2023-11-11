import cppyy
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Modified from https://archive.ics.uci.edu/dataset/53/iris
# Fetch iris dataset from UCI repository.
iris = fetch_ucirepo(id=53) 

# Features and labels.
# Features is a numpy array of shape (150, 4).
features = iris.data.features.values
# Labels is a numpy array of shape (150, 1).
labels = iris.data.targets.values.ravel()

# Map labels to integers.
# Sklearn's LabelEncoder is used to encode labels with value between 0 and n_classes-1.
# This effectively maps the labels to integers as so:
# 'Iris-setosa' -> 0
# 'Iris-versicolor' -> 1
# 'Iris-virginica' -> 2
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data 50-50 into training and validation sets.
# train_test_split is a convenience function to split data into training and validation sets.
training_features, validation_features, training_labels, validation_labels = train_test_split(features, encoded_labels, test_size=0.5, random_state=0)

# Bind the C++ library.
cppyy.include('./include/fully_connected_nn.hpp')
cppyy.include('./include/loss_function.hpp')
cppyy.include('./include/sgd.hpp')
cppyy.load_library('./lib/lib_fully_connected_nn.dll')

def train_model(model: cppyy.gbl.FullyConnectedNN, loss_function: cppyy.gbl.LossFunction, optimizer: cppyy.gbl.SGD) -> None:
    MAX_EPOCHS = 1000
    
    best_avg_testing_loss = np.inf

    for epoch in range(MAX_EPOCHS):
        # Training.
        training_loss = 0.0
        for i in range(len(training_features)):
            output = model(training_features[i].tolist())
            training_loss += loss_function(output, int(training_labels[i]))

            print(f'Output: {output} | Target: {training_labels[i]}')

            loss_gradient = loss_function.derivative(output, training_labels[i].tolist())
            model.backward(loss_gradient)

            #optimizer.step(model)

        avg_training_loss = training_loss / len(training_features)
        print('ran')

        # Validation.
        testing_loss = 0.0
        for i in range(len(validation_features)):
            output = model(validation_features[i].tolist())
            testing_loss += loss_function(output, int(validation_labels[i]))

        avg_testing_loss = testing_loss / len(validation_features)

        if avg_testing_loss < best_avg_testing_loss:
            best_avg_testing_loss = avg_testing_loss

        print(f'Epoch {epoch + 1}: Training loss: {avg_training_loss} | Testing loss: {avg_testing_loss}')


model_one = cppyy.gbl.FullyConnectedNN()
model_one.insert_layer(cppyy.gbl.Tanh(4, 2))
model_one.insert_layer(cppyy.gbl.Softmax(2, 3))

model_two = cppyy.gbl.FullyConnectedNN()
model_two.insert_layer(cppyy.gbl.ReLU(4, 6))
model_two.insert_layer(cppyy.gbl.ReLU(6, 3))

loss_function = cppyy.gbl.CrossEntropyLoss()
optimizer = cppyy.gbl.SGD(0.01, 0.9, 1e-8)

train_model(model_one, loss_function, optimizer)

