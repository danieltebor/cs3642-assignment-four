import tkinter as tk

from model import *

window = tk.Tk()
window.title('CS3642 Assignment 4')
window.geometry('645x600')


### Layer Config Frame ###
layer_config_frame = tk.Frame(window, relief='groove', borderwidth=2)
layer_config_frame.grid(row=0, column=0, sticky='nsew')

# Configure column weights.
layer_config_frame.columnconfigure(0, weight=1)
layer_config_frame.columnconfigure(1, weight=1)

# Layer config label.
layer_config_label = tk.Label(layer_config_frame, text='Add Layer', relief='groove')
layer_config_label.grid(row=0, column=0, columnspan=2, sticky='nsew')

def validate_integer_input(input):
    if input.isdigit() or input == '':
        return True
    return False
validate_integer_input_command = window.register(validate_integer_input)

# Input size input.
layer_input_size_label = tk.Label(layer_config_frame, text='Input Size', relief='groove')
layer_input_size_label.grid(row=1, column=0, sticky='nsew')
layer_input_size_entry = tk.Entry(layer_config_frame, validate='key', validatecommand=(validate_integer_input_command, '%P'))
layer_input_size_entry.grid(row=1, column=1, sticky='nsew')
layer_input_size_entry.insert(0, '4')
layer_input_size_entry.config(state='readonly')

# Output size input.
layer_output_size_label = tk.Label(layer_config_frame, text='Output Size', relief='groove')
layer_output_size_label.grid(row=2, column=0, sticky='nsew')
layer_output_size_entry = tk.Entry(layer_config_frame, validate='key', validatecommand=(validate_integer_input_command, '%P'))
layer_output_size_entry.grid(row=2, column=1, sticky='nsew')

# Activation function selection.
ACTIVATION_FUNCTIONS = ['Linear', 'Sigmoid', 'ReLU', 'Softmax']
layer_activation_var = tk.StringVar(window)
layer_activation_var.set(ACTIVATION_FUNCTIONS[1])
layer_activation_label = tk.Label(layer_config_frame, text='Activation Function', relief='groove')
layer_activation_label.grid(row=3, column=0, sticky='nsew')
layer_activation_option = tk.OptionMenu(layer_config_frame, layer_activation_var , *ACTIVATION_FUNCTIONS)
layer_activation_option.grid(row=3, column=1, sticky='nsew')

def reset_layer_config():
    global layer_input_size_entry
    global layer_output_size_entry
    global layer_error_label

    layer_error_label.config(text='')

    layer_input_size_entry.config(state='normal')
    layer_input_size_entry.delete(0, tk.END)
    layer_input_size_entry.insert(0, layer_output_size_entry.get())
    layer_input_size_entry.config(state='readonly')
    layer_output_size_entry.delete(0, tk.END)

# Add layer error label.
layer_error_label = tk.Label(layer_config_frame, text='', relief='groove', fg='red')
layer_error_label.grid(row=6, column=0, columnspan=2, sticky='nsew')


### Model Layers Frame ###
model_layers_frame = tk.Frame(window, relief='groove', borderwidth=2)
model_layers_frame.grid(row=1, column=0, sticky='nsew')

# Configure column weights.
model_layers_frame.columnconfigure(0, weight=1)
model_layers_frame.columnconfigure(1, weight=1)

# Model layers label.
model_layers_label = tk.Label(model_layers_frame, text='Model Layers', relief='groove')
model_layers_label.grid(row=0, column=0, columnspan=2, sticky='nsew')

# Model layers header labels.
model_layers_header_shape_label = tk.Label(model_layers_frame, text='Shape', relief='groove')
model_layers_header_shape_label.grid(row=2, column=0, sticky='nsew')
model_layers_header_activation_label = tk.Label(model_layers_frame, text='Activation Function', relief='groove')
model_layers_header_activation_label.grid(row=2, column=1, sticky='nsew')

# Add layer button.
layers_info = []

def update_layers_info():
    global layers_info
    global layer_error_label

    if len(layers_info) > 0 and layers_info[-1][2] == 'Softmax':
        layer_error_label.config(text='Cannot add layers after softmax.')
        return

    input_size = int(layer_input_size_entry.get())
    if layer_output_size_entry.get() == '':
        layer_error_label.config(text='Output size cannot be empty.')
        return
    output_size = int(layer_output_size_entry.get())
    activation_function = layer_activation_var.get()

    row = len(layers_info) + 3
    layers_info.append((input_size, output_size, activation_function))

    # Shape label.
    layer_shape_label = tk.Label(model_layers_frame, text=f'({input_size}, {output_size})', relief='groove')
    layer_shape_label.grid(row=row, column=0, sticky='nsew')

    # Activation function label.
    layer_activation_label = tk.Label(model_layers_frame, text=activation_function, relief='groove')
    layer_activation_label.grid(row=row, column=1, sticky='nsew')

    reset_layer_config()

add_layer_button = tk.Button(layer_config_frame, text='Add Layer', command=update_layers_info, relief='raised', borderwidth=2)
add_layer_button.grid(row=4, column=0, columnspan=2, sticky='nsew')

# Reset layers button.
def reset_layers():
    global layers_info
    global layer_output_size_entry
    
    layers_info = []
    for widget in model_layers_frame.winfo_children():
        if widget not in [model_layers_label, model_layers_header_shape_label, model_layers_header_activation_label]:
            widget.destroy()

    layer_output_size_entry.delete(0, tk.END)
    layer_output_size_entry.insert(0, '4')

    reset_layer_config()

reset_layers_button = tk.Button(layer_config_frame, text='Reset Layers', command=reset_layers, relief='raised', borderwidth=2)
reset_layers_button.grid(row=5, column=0, columnspan=2, sticky='nsew')


### Model Config Frame ###
model_config_frame = tk.Frame(window, relief='groove', borderwidth=2)
model_config_frame.grid(row=0, column=1, sticky='nsew')

# Model config label.
model_config_label = tk.Label(model_config_frame, text='Model Configuration', relief='groove')
model_config_label.grid(row=0, column=0, columnspan=2, sticky='nsew')

def validate_float_input(input):
    if input.isdigit() or input == '':
        return True
    try:
        float(input)
        return True
    except ValueError:
        return False
validate_float_input_command = window.register(validate_float_input)

# Learning rate input.
model_learning_rate_label = tk.Label(model_config_frame, text='Learning Rate', relief='groove')
model_learning_rate_label.grid(row=1, column=0, sticky='nsew')
model_learning_rate_entry = tk.Entry(model_config_frame, validate='key', validatecommand=(validate_float_input_command, '%P'))
model_learning_rate_entry.grid(row=1, column=1, sticky='nsew')
model_learning_rate_entry.insert(0, '0.01')

# Momentum input.
model_momentum_label = tk.Label(model_config_frame, text='Momentum', relief='groove')
model_momentum_label.grid(row=2, column=0, sticky='nsew')
model_momentum_entry = tk.Entry(model_config_frame, validate='key', validatecommand=(validate_float_input_command, '%P'))
model_momentum_entry.grid(row=2, column=1, sticky='nsew')
model_momentum_entry.insert(0, '0.0')

# Weight decay input.
model_weight_decay_label = tk.Label(model_config_frame, text='Weight Decay', relief='groove')
model_weight_decay_label.grid(row=3, column=0, sticky='nsew')
model_weight_decay_entry = tk.Entry(model_config_frame, validate='key', validatecommand=(validate_float_input_command, '%P'))
model_weight_decay_entry.grid(row=3, column=1, sticky='nsew')
model_weight_decay_entry.insert(0, '0.0')

# Model config error label.
model_config_error_label = tk.Label(model_config_frame, text='', relief='groove', fg='red')
model_config_error_label.grid(row=5, column=0, columnspan=2, sticky='nsew')

# Training metadata labels.
training_metadata_label = tk.Label(model_config_frame, text='Training Metadata', relief='groove')
training_metadata_label.grid(row=6, column=0, columnspan=2, sticky='nsew')

epochs_taken_label = tk.Label(model_config_frame, text='Epochs Taken', relief='groove')
epochs_taken_label.grid(row=7, column=0, sticky='nsew')
epochs_taken_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
epochs_taken_value_label.grid(row=7, column=1, sticky='nsew')

loss_function_label = tk.Label(model_config_frame, text='Loss Function', relief='groove')
loss_function_label.grid(row=8, column=0, sticky='nsew')
loss_function_value_label = tk.Label(model_config_frame, text='-', relief='ridge', anchor='w')
loss_function_value_label.grid(row=8, column=1, sticky='nsew')

starting_avg_training_loss_label = tk.Label(model_config_frame, text='Starting Avg Training Loss', relief='groove')
starting_avg_training_loss_label.grid(row=9, column=0, sticky='nsew')
starting_avg_training_loss_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
starting_avg_training_loss_value_label.grid(row=9, column=1, sticky='nsew')

best_avg_training_loss_label = tk.Label(model_config_frame, text='Best Avg Training Loss', relief='groove')
best_avg_training_loss_label.grid(row=10, column=0, sticky='nsew')
best_avg_training_loss_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
best_avg_training_loss_value_label.grid(row=10, column=1, sticky='nsew')

starting_avg_testing_loss_label = tk.Label(model_config_frame, text='Starting Avg Testing Loss', relief='groove')
starting_avg_testing_loss_label.grid(row=11, column=0, sticky='nsew')
starting_avg_testing_loss_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
starting_avg_testing_loss_value_label.grid(row=11, column=1, sticky='nsew')

best_avg_testing_loss_label = tk.Label(model_config_frame, text='Best Avg Testing Loss', relief='groove')
best_avg_testing_loss_label.grid(row=12, column=0, sticky='nsew')
best_avg_testing_loss_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
best_avg_testing_loss_value_label.grid(row=12, column=1, sticky='nsew')

accuracy_label = tk.Label(model_config_frame, text='Accuracy', relief='groove')
accuracy_label.grid(row=13, column=0, sticky='nsew')
accuracy_value_label = tk.Label(model_config_frame, text='0', relief='ridge', anchor='w')
accuracy_value_label.grid(row=13, column=1, sticky='nsew')

# Train model button.
model = None

def _train_model():
    global model
    global model_config_error_label
    global epochs_taken_value_label
    global loss_function_value_label
    global starting_avg_training_loss_value_label
    global best_avg_training_loss_value_label
    global starting_avg_testing_loss_value_label
    global best_avg_testing_loss_value_label
    global accuracy_value_label

    model_config_error_label.config(text='')

    if len(layers_info) < 1:
        model_config_error_label.config(text='Must have at least one layer.')
        return
    elif layers_info[-1][1] != 3:
        model_config_error_label.config(text='Output layer must have an output size of 3.')
        return

    layer_sizes = []
    activation_functions = []

    for layer_info in layers_info:
        layer_sizes.append(layer_info[0])
        activation_functions.append(layer_info[2].lower())
    layer_sizes.append(layers_info[-1][1])

    model = create_model(layer_sizes, activation_functions)

    loss_function = cppyy.gbl.MSELoss()
    loss_function_value_label.config(text='MSE Loss')
    if layers_info[-1][2] == 'Softmax':
        loss_function = cppyy.gbl.CrossEntropyLoss()
        loss_function_value_label.config(text='Cross Entropy Loss')

    if model_learning_rate_entry.get() == '':
        model_config_error_label.config(text='Learning rate cannot be empty.')
        return
    learning_rate = float(model_learning_rate_entry.get())
    if model_momentum_entry.get() == '':
        model_config_error_label.config(text='Momentum cannot be empty.')
        return
    momentum = float(model_momentum_entry.get())
    if model_weight_decay_entry.get() == '':
        model_config_error_label.config(text='Weight decay cannot be empty.')
        return
    weight_decay = float(model_weight_decay_entry.get())

    optimizer = create_optimizer(learning_rate, momentum, weight_decay)

    training_metadata = train_model(model, loss_function, optimizer)
    accuracy = calc_accuracy(model) * 100

    epochs_taken_value_label.config(text=f'{training_metadata.epochs_taken}')
    starting_avg_training_loss_value_label.config(text=f'{training_metadata.starting_avg_training_loss:.4f}')
    best_avg_training_loss_value_label.config(text=f'{training_metadata.best_avg_training_loss:.4f}')
    starting_avg_testing_loss_value_label.config(text=f'{training_metadata.starting_avg_testing_loss:.4f}')
    best_avg_testing_loss_value_label.config(text=f'{training_metadata.best_avg_testing_loss:.4f}')
    accuracy_value_label.config(text=f'{accuracy:.4f}%')

train_model_button = tk.Button(model_config_frame, text='Train Model', command=_train_model, relief='raised', borderwidth=2)
train_model_button.grid(row=4, column=0, columnspan=2, sticky='nsew')


### Model Prediction Frame ###
model_prediction_frame = tk.Frame(window, relief='groove', borderwidth=2)
model_prediction_frame.grid(row=1, column=1, sticky='nsew')

# Configure column weights.
model_prediction_frame.columnconfigure(0, weight=1)
model_prediction_frame.columnconfigure(1, weight=1)

# Model prediction label.
model_prediction_label = tk.Label(model_prediction_frame, text='Model Prediction', relief='groove')
model_prediction_label.grid(row=0, column=0, columnspan=2, sticky='nsew')

# Model input.
model_prediction_features_label = tk.Label(model_prediction_frame, text='Features', relief='groove')
model_prediction_features_label.grid(row=1, column=0, sticky='nsew')
model_prediction_features_value_label = tk.Label(model_prediction_frame, text='[-, -, -, -]', relief='ridge', anchor='w')
model_prediction_features_value_label.grid(row=1, column=1, sticky='nsew')

# Model expected output.
model_prediction_expected_output_label = tk.Label(model_prediction_frame, text='Expected Output', relief='groove')
model_prediction_expected_output_label.grid(row=2, column=0, sticky='nsew')
model_prediction_expected_output_value_label = tk.Label(model_prediction_frame, text='-', relief='ridge', anchor='w')
model_prediction_expected_output_value_label.grid(row=2, column=1, sticky='nsew')

# Model predicted output.
model_prediction_predicted_output_label = tk.Label(model_prediction_frame, text='Predicted Output', relief='groove')
model_prediction_predicted_output_label.grid(row=3, column=0, sticky='nsew')
model_prediction_predicted_output_value_label = tk.Label(model_prediction_frame, text='-', relief='ridge', anchor='w')
model_prediction_predicted_output_value_label.grid(row=3, column=1, sticky='nsew')

# Model prediction error label.
model_prediction_error_label = tk.Label(model_prediction_frame, text='', relief='groove', fg='red')
model_prediction_error_label.grid(row=4, column=0, columnspan=2, sticky='nsew')

# Model prediction shuffle button.
def shuffle():
    global model_prediction_features_value_label
    global model_prediction_expected_output_value_label
    global model_prediction_predicted_output_value_label
    global model_prediction_error_label

    model_prediction_error_label.config(text='')

    if model is None:
        model_prediction_error_label.config(text='Must train a model first.')
        return

    np.random.shuffle(VALIDATION_DATA)

    feature, expected_label = VALIDATION_DATA[0]
    feature = feature.tolist()
    expected_label = label_encoder.inverse_transform([expected_label])[0]

    model_prediction_features_value_label.config(text=f'{feature}')
    model_prediction_expected_output_value_label.config(text=f'{expected_label}')

    predicted_output = model.predict(feature)
    predicted_output = label_encoder.inverse_transform([predicted_output])[0]
    model_prediction_predicted_output_value_label.config(text=f'{predicted_output}')

shuffle_button = tk.Button(model_prediction_frame, text='Shuffle', command=shuffle, relief='raised', borderwidth=2)
shuffle_button.grid(row=5, column=0, columnspan=2, sticky='nsew')

window.mainloop()