import math

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
print(training_labels)

# Bind the C++ library.
cppyy.include('./include/fully_connected_nn.hpp')
cppyy.include('./include/loss_function.hpp')
cppyy.include('./include/sgd.hpp')
cppyy.load_library('./lib/lib_fully_connected_nn.dll')

def train_model(model: cppyy.gbl.FullyConnectedNN, loss_function: cppyy.gbl.LossFunction, optimizer: cppyy.gbl.SGD) -> None:
    MAX_EPOCHS = 1000
    MAX_FORGIVES = 4
    times_forgiven = 0
    best_avg_testing_loss = np.inf

    for epoch in range(MAX_EPOCHS):
        # Train the network by doing a forward and backward pass for each input.
        # Run SGD to adjust model weights.
        avg_training_loss = 0.0
        
        for i in range(len(training_features)):
            # Get output from forward pass.
            output = model(training_features[i].tolist())
            avg_training_loss += loss_function(output, int(training_labels[i]))
            
            # Compute loss gradient and perform backward pass.
            loss_gradient = loss_function.derivative(output, int(training_labels[i]))
            model.backward(loss_gradient)
            
            # Adjust weights. 
            optimizer.step(model)
            
        avg_training_loss /= len(training_features)
        
        # Test the loss of the network to quantify improvement.
        avg_testing_loss = 0.0
        
        for i in range(len(validation_features)):
            # Get output from forward pass
            output = model(validation_features[i].tolist())
            avg_testing_loss += loss_function(output, int(validation_labels[i]))
            
        avg_testing_loss /= len(validation_features)
        
        print(f'[Epoch {epoch + 1}/{MAX_EPOCHS}] Avg Training Loss: {avg_training_loss:.4f} '
              + '| Avg Testing Loss: {avg_testing_loss:.4f}')

        if avg_testing_loss < best_avg_testing_loss:
            best_avg_testing_loss = avg_testing_loss
            times_forgiven = 0
        else:
            times_forgiven += 1
            
        if times_forgiven > MAX_FORGIVES:
            print(f'Model has gone {times_forgiven} epochs with not improvement. Halting training')
            break
            

model_one = cppyy.gbl.FullyConnectedNN()
model_one.insert_layer(cppyy.gbl.ReLU(4, 2))
model_one.insert_layer(cppyy.gbl.Softmax(2, 3))

model_two = cppyy.gbl.FullyConnectedNN()
model_two.insert_layer(cppyy.gbl.ReLU(4, 6))
model_two.insert_layer(cppyy.gbl.Softmax(6, 3))

loss_function = cppyy.gbl.CrossEntropyLoss()

optimizer_one = cppyy.gbl.SGD(0.01)
optimizer_two = cppyy.gbl.SGD(0.01)

train_model(model=model_one, loss_function=loss_function, optimizer=optimizer_one)
train_model(model=model_two, loss_function=loss_function, optimizer=optimizer_two)

accuracy_one = 0
accuracy_two = 0
for input, label in zip(validation_features, validation_labels):
    if model_one.predict(input.tolist()) == int(label):
        accuracy_one += 1
        
    if model_two.predict(input.tolist()) == int(label):
        accuracy_two += 1
        
accuracy_one /= len(validation_features)
accuracy_two /= len(validation_features)

print(accuracy_one)
print(accuracy_two)