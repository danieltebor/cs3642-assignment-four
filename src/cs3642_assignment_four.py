import cppyy
import numpy as np

from iris_dataset import *

# Bind the C++ library.
cppyy.include('./include/fully_connected_nn.hpp')
cppyy.include('./include/loss_function.hpp')
cppyy.include('./include/sgd.hpp')
cppyy.load_library('./lib/lib_fully_connected_nn.dll')

def train_model(model: cppyy.gbl.FullyConnectedNN, loss_function: cppyy.gbl.LossFunction, optimizer: cppyy.gbl.SGD) -> None:
    MAX_EPOCHS = 1000
    MAX_FORGIVES = 9
    times_forgiven = 0
    best_avg_testing_loss = np.inf

    for epoch in range(MAX_EPOCHS):
        # Train the network by doing a forward and backward pass for each input.
        # Run SGD to adjust model weights.
        avg_training_loss = 0.0
        
        for i in range(len(TRAINING_FEATURES)):
            # Get output from forward pass.
            output = model(TRAINING_FEATURES[i].tolist())
            avg_training_loss += loss_function(output, int(TRAINING_LABELS[i]))
            
            # Compute loss gradient and perform backward pass.
            model.backward(int(TRAINING_LABELS[i]))
            
            #print(model.get_layers()[0].get_biases())

            # Adjust weights. 
            optimizer.step(model)
            
        avg_training_loss /= len(TRAINING_FEATURES)
        
        # Test the loss of the network to quantify improvement.
        avg_testing_loss = 0.0
        
        for i in range(len(VALIDATION_FEATURES)):
            # Get output from forward pass
            output = model(VALIDATION_FEATURES[i].tolist())
            avg_testing_loss += loss_function(output, int(VALIDATION_LABELS[i]))
            
        avg_testing_loss /= len(VALIDATION_FEATURES)
        
        print(f'[Epoch {epoch + 1}/{MAX_EPOCHS}] Avg Training Loss: {avg_training_loss:.4f} '
              + f'| Avg Testing Loss: {avg_testing_loss:.4f}')

        if avg_testing_loss < best_avg_testing_loss:
            best_avg_testing_loss = avg_testing_loss
            times_forgiven = 0
        else:
            times_forgiven += 1
            
        if times_forgiven > MAX_FORGIVES:
            print(f'Model has gone {times_forgiven} epochs without improvement. Halting training')
            break
            

model_one = cppyy.gbl.FullyConnectedNN()
model_one.insert_layer(cppyy.gbl.Sigmoid(4, 2))
model_one.insert_layer(cppyy.gbl.Sigmoid(2, 3))

model_two = cppyy.gbl.FullyConnectedNN()
model_two.insert_layer(cppyy.gbl.Sigmoid(4, 6))
model_two.insert_layer(cppyy.gbl.Sigmoid(6, 3))

loss_function = cppyy.gbl.MeanSquaredError()

optimizer_one = cppyy.gbl.SGD(0.1)
optimizer_two = cppyy.gbl.SGD(0.1)

train_model(model=model_one, loss_function=loss_function, optimizer=optimizer_one)
train_model(model=model_two, loss_function=loss_function, optimizer=optimizer_two)

print('Model One Predictions:')
for i in range(len(VALIDATION_FEATURES)):
    output = model_one.predict(VALIDATION_FEATURES[i].tolist())
    print(f'Predicted: {output} | Actual: {int(VALIDATION_LABELS[i])}')

print('Model Two Predictions:')
for i in range(len(VALIDATION_FEATURES)):
    output = model_two.predict(VALIDATION_FEATURES[i].tolist())
    print(f'Predicted: {output} | Actual: {int(VALIDATION_LABELS[i])}')