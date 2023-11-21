import time

from model import *

avg_model_one_accuracy = 0.0
avg_model_two_accuracy = 0.0
avg_model_one_avg_epoch_runtime = 0.0
avg_model_two_avg_epoch_runtime = 0.0

for i in range(10000):
    model_one = cppyy.gbl.FullyConnectedNN()
    model_one.insert_layer(cppyy.gbl.ReLU(4, 2))
    model_one.insert_layer(cppyy.gbl.Softmax(2, 3))

    model_two = cppyy.gbl.FullyConnectedNN()
    model_two.insert_layer(cppyy.gbl.ReLU(4, 6))
    model_two.insert_layer(cppyy.gbl.Softmax(6, 3))

    loss_function = cppyy.gbl.CrossEntropyLoss()

    optimizer = cppyy.gbl.SGD(0.005, 0.3, 0.0001)

    start_time = time.time_ns()
    model_one_training_metadata = train_model(model_one, loss_function, optimizer)
    end_time = time.time_ns()
    model_one_avg_epoch_runtime = (end_time - start_time) / model_one_training_metadata.epochs_taken

    start_time = time.time_ns()
    model_two_training_metadata = train_model(model_two, loss_function, optimizer)
    end_time = time.time_ns()
    model_two_avg_epoch_runtime = (end_time - start_time) / model_two_training_metadata.epochs_taken

    avg_model_one_avg_epoch_runtime += model_one_avg_epoch_runtime
    avg_model_two_avg_epoch_runtime += model_two_avg_epoch_runtime

    model_one_accuracy = calc_accuracy(model_one)
    model_two_accuracy = calc_accuracy(model_two)

    avg_model_one_accuracy += model_one_accuracy
    avg_model_two_accuracy += model_two_accuracy

    if i % 100 == 0:
        print(f'Iteration {i}')
        print(f'Model One Accuracy: {model_one_accuracy:.4f}')
        print(f'Model Two Accuracy: {model_two_accuracy:.4f}')
        print(f'Model One Avg Epoch Runtime: {model_one_avg_epoch_runtime:.4f}')
        print(f'Model Two Avg Epoch Runtime: {model_two_avg_epoch_runtime:.4f}')
        print()

avg_model_one_avg_epoch_runtime /= 10000
avg_model_two_avg_epoch_runtime /= 10000
avg_model_one_accuracy /= 10000
avg_model_two_accuracy /= 10000

print(f'Average Model One Accuracy: {avg_model_one_accuracy:.4f}')
print(f'Average Model Two Accuracy: {avg_model_two_accuracy:.4f}')
print(f'Average Model One Avg Epoch Runtime: {avg_model_one_avg_epoch_runtime:.4f}')
print(f'Average Model Two Avg Epoch Runtime: {avg_model_two_avg_epoch_runtime:.4f}')