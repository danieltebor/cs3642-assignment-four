import cppyy
import numpy as np

from iris_dataset import *

# Bind the C++ library.
cppyy.include('./include/fully_connected_nn.hpp')
cppyy.include('./include/loss_function.hpp')
cppyy.include('./include/sgd.hpp')
cppyy.load_library('./lib/lib_fully_connected_nn.dll')

class TrainingMetadata:
    def __init__(self, epochs_taken: float,
                 starting_avg_training_loss: float, best_avg_training_loss: float,
                 starting_avg_testing_loss: float, best_avg_testing_loss: float):
        self.epochs_taken = epochs_taken
        self.starting_avg_training_loss = starting_avg_training_loss
        self.best_avg_training_loss = best_avg_training_loss
        self.starting_avg_testing_loss = starting_avg_testing_loss
        self.best_avg_testing_loss = best_avg_testing_loss

    def __str__(self) -> str:
        return f'Epochs Taken: {self.epochs_taken}\n' \
               + f'Starting Avg Training Loss: {self.starting_avg_training_loss:.4f}\n' \
               + f'Best Avg Training Loss: {self.best_avg_training_loss:.4f}\n' \
               + f'Starting Avg Testing Loss: {self.starting_avg_testing_loss:.4f}\n' \
               + f'Best Avg Testing Loss: {self.best_avg_testing_loss:.4f}'

def create_model(layer_sizes: list[int], activation_functions: list[str]) -> cppyy.gbl.FullyConnectedNN:
    if len(layer_sizes) < 2:
        raise ValueError('Number of layers must be at least 2.')
    elif layer_sizes[0] != 4:
        raise ValueError('Input layer must have 4 nodes.')
    elif layer_sizes[-1] != 3:
        raise ValueError('Output layer must have 3 nodes.')
    if len(layer_sizes) != len(activation_functions) + 1:
        raise ValueError('Number of layers must equal number of activation functions + 1.')
    
    # Create model.
    model = cppyy.gbl.FullyConnectedNN()
    
    # Insert layers.
    for i in range(len(activation_functions)):
        if activation_functions[i] == 'linear':
            model.insert_layer(cppyy.gbl.Linear(layer_sizes[i], layer_sizes[i + 1]))
        elif activation_functions[i] == 'sigmoid':
            model.insert_layer(cppyy.gbl.Sigmoid(layer_sizes[i], layer_sizes[i + 1]))
        elif activation_functions[i] == 'relu':
            model.insert_layer(cppyy.gbl.ReLU(layer_sizes[i], layer_sizes[i + 1]))
        elif activation_functions[i] == 'softmax':
            model.insert_layer(cppyy.gbl.Softmax(layer_sizes[i], layer_sizes[i + 1]))
        else:
            raise ValueError(f'Invalid activation function: {activation_functions[i]}')
            
    return model

def create_optimizer(learning_rate: float, momentum: float, weight_decay: float) -> cppyy.gbl.SGD:
    return cppyy.gbl.SGD(learning_rate, momentum, weight_decay)

def train_model(model: cppyy.gbl.FullyConnectedNN,
                loss_function: cppyy.gbl.LossFunction,
                optimizer: cppyy.gbl.SGD) -> (float, float, float, float):
    MAX_EPOCHS = 10000
    MAX_FORGIVES = 4
    times_forgiven = 0
    starting_avg_training_loss = np.inf
    best_avg_training_loss = np.inf
    starting_avg_testing_loss = np.inf
    best_avg_testing_loss = np.inf

    for epoch in range(MAX_EPOCHS):
        # Train the network by doing a forward and backward pass for each input.
        # Run SGD to adjust model weights.

        # Shuffle training data.
        np.random.shuffle(TRAINING_DATA)

        avg_training_loss = 0.0
        
        for feature, label in TRAINING_DATA:
            # Get output from forward pass.
            output = model(feature.tolist())
            avg_training_loss += loss_function(output, int(label))
            
            # Compute loss gradient and perform backward pass.
            model.backward(int(label))
            
            #print(model.get_layers()[0].get_biases())

            # Adjust weights. 
            optimizer.step(model)
            
        avg_training_loss /= len(TRAINING_DATA)
        
        # Test the loss of the network to quantify improvement.
        avg_testing_loss = 0.0
        
        for feature, label in VALIDATION_DATA:
            # Get output from forward pass
            output = model(feature.tolist())
            avg_testing_loss += loss_function(output, int(label))
            
        avg_testing_loss /= len(VALIDATION_DATA)
        
        #print(f'[Epoch {epoch + 1}/{MAX_EPOCHS}] Avg Training Loss: {avg_training_loss:.4f} '
        #      + f'| Avg Testing Loss: {avg_testing_loss:.4f}')

        if epoch == 0:
            starting_avg_training_loss = avg_training_loss
            starting_avg_testing_loss = avg_testing_loss

        if avg_training_loss < best_avg_training_loss:
            best_avg_training_loss = avg_training_loss

        if avg_testing_loss < best_avg_testing_loss:
            best_avg_testing_loss = avg_testing_loss
            times_forgiven = 0
        else:
            times_forgiven += 1
            
        if times_forgiven > MAX_FORGIVES:
            #print(f'Model has gone {times_forgiven} epochs without improvement. Halting training')
            break

    return TrainingMetadata(epoch + 1, starting_avg_training_loss, best_avg_training_loss, starting_avg_testing_loss, best_avg_testing_loss)

def calc_accuracy(model: cppyy.gbl.FullyConnectedNN) -> float:
    correct = 0
    
    for feature, label in VALIDATION_DATA:
        output = model(feature.tolist())
        prediction = np.argmax(output)
        
        if prediction == int(label):
            correct += 1
            
    return correct / len(VALIDATION_DATA)